"""
Download Manager for HuggingFace Hub downloads with progress tracking.

Uses multiprocessing for downloads so they can be forcibly terminated on cancellation.
Progress updates are communicated via multiprocessing.Queue.
"""

import uuid
import threading
import time
import os
import glob
from queue import Empty
from multiprocessing import Process, Queue
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Callable
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from core.common import PRNT_API


def _cleanup_incomplete_download(cache_dir: str, repo_id: str, filename: str) -> None:
    """
    Clean up incomplete HuggingFace Hub download files.

    When a download is cancelled, partial files and lock files may remain:
    - .incomplete files (partial downloads)
    - .lock files (download locks)

    This function removes these artifacts from the cache directory.
    """
    if not cache_dir or not repo_id:
        return

    try:
        # HuggingFace cache structure: <cache_dir>/models--<org>--<repo>/
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        repo_cache_path = os.path.join(cache_dir, repo_cache_name)

        if not os.path.exists(repo_cache_path):
            return

        # Find and remove .incomplete files
        incomplete_pattern = os.path.join(repo_cache_path, "**", "*.incomplete")
        for incomplete_file in glob.glob(incomplete_pattern, recursive=True):
            try:
                os.remove(incomplete_file)
                print(
                    f"{PRNT_API} Removed incomplete file: {incomplete_file}", flush=True
                )
            except Exception as e:
                print(f"{PRNT_API} Could not remove {incomplete_file}: {e}", flush=True)

        # Find and remove .lock files
        lock_pattern = os.path.join(repo_cache_path, "**", "*.lock")
        for lock_file in glob.glob(lock_pattern, recursive=True):
            try:
                os.remove(lock_file)
                print(f"{PRNT_API} Removed lock file: {lock_file}", flush=True)
            except Exception as e:
                print(f"{PRNT_API} Could not remove {lock_file}: {e}", flush=True)

    except Exception as e:
        print(f"{PRNT_API} Error cleaning up incomplete download: {e}", flush=True)


@dataclass
class DownloadProgress:
    """Represents the current state of a download task."""

    task_id: str
    repo_id: str
    filename: str
    cache_dir: str = ""  # Cache directory for cleanup on cancel
    downloaded_bytes: int = 0
    total_bytes: int = 0
    percent: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: Optional[int] = None
    status: str = "pending"  # pending, downloading, completed, error, cancelled
    error: Optional[str] = None
    file_path: Optional[str] = None  # Set on completion
    secondary_task_id: Optional[str] = None  # For chained downloads (e.g., mmproj)
    completed_at: Optional[float] = None  # Timestamp for auto-cleanup

    def to_dict(self) -> dict:
        return asdict(self)


class ProgressCaptureTqdm(tqdm):
    """
    Custom tqdm class that captures progress and sends to a queue.
    Used to intercept huggingface_hub download progress in subprocess.
    """

    def __init__(
        self,
        *args,
        progress_queue: Queue = None,
        task_id: str = None,
        **kwargs,
    ):
        self._progress_queue = progress_queue
        self._task_id = task_id
        self._last_update_time = time.time()
        self._update_interval = 0.2  # Minimum seconds between updates
        # Filter out huggingface-hub v1.x XET-specific arguments that tqdm doesn't recognize
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

    def update(self, n: int = 1) -> bool | None:
        result = super().update(n)

        # Throttle progress updates to avoid overwhelming the queue
        current_time = time.time()
        if current_time - self._last_update_time < self._update_interval:
            return result

        self._last_update_time = current_time

        if self._progress_queue and self.total:
            # Calculate speed in MB/s
            rate = self.format_dict.get("rate", 0) or 0
            speed_mbps = rate / (1024 * 1024) if rate else 0.0

            # Calculate ETA
            eta = None
            if rate and rate > 0:
                remaining = self.total - self.n
                eta = int(remaining / rate)

            try:
                self._progress_queue.put_nowait(
                    {
                        "type": "progress",
                        "task_id": self._task_id,
                        "downloaded_bytes": self.n,
                        "total_bytes": self.total,
                        "percent": (self.n / self.total * 100) if self.total else 0,
                        "speed_mbps": round(speed_mbps, 2),
                        "eta_seconds": eta,
                    }
                )
            except Exception:
                pass  # Queue full, skip this update
        return result


def _download_worker(
    task_id: str,
    repo_id: str,
    filename: str,
    cache_dir: str,
    progress_queue: Queue,
):
    """
    Download worker function that runs in a separate process.
    Sends progress updates and results via the queue.
    Can be terminated at any time without cleanup issues.
    """
    try:
        # Create factory for tqdm class with bound queue
        def make_tqdm_class():
            class BoundProgressTqdm(ProgressCaptureTqdm):
                def __init__(self, *args, **kwargs):
                    kwargs["progress_queue"] = progress_queue
                    kwargs["task_id"] = task_id
                    super().__init__(*args, **kwargs)

            return BoundProgressTqdm

        # Send downloading status
        progress_queue.put(
            {
                "type": "status",
                "task_id": task_id,
                "status": "downloading",
            }
        )

        # Perform the download
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            tqdm_class=make_tqdm_class(),
        )

        # Send completion
        progress_queue.put(
            {
                "type": "complete",
                "task_id": task_id,
                "file_path": file_path,
            }
        )

    except Exception as e:
        # Send error
        progress_queue.put(
            {
                "type": "error",
                "task_id": task_id,
                "error": str(e),
            }
        )


class DownloadManager:
    """
    Manages concurrent HuggingFace model downloads with progress tracking.

    Features:
    - Non-blocking downloads using multiprocessing
    - Real-time progress tracking via queue
    - Reliable cancellation via process termination
    - Thread-safe progress state
    - Automatic cleanup of completed tasks
    """

    # Time in seconds to keep completed tasks before auto-cleanup
    TASK_RETENTION_SECONDS = 300  # 5 minutes

    def __init__(self, max_workers: int = 3):
        self._max_workers = max_workers
        self._progress: Dict[str, DownloadProgress] = {}
        self._processes: Dict[str, Process] = {}
        self._queues: Dict[str, Queue] = {}
        self._callbacks: Dict[str, dict] = {}  # Store callbacks per task
        self._lock = threading.Lock()
        self._shutdown_flag = threading.Event()

        # Shared queue for all progress updates
        self._progress_queue = Queue()

        # Start background thread to process queue updates
        self._queue_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._queue_thread.start()

    def _process_queue(self):
        """Background thread that processes progress updates from all download processes."""
        while not self._shutdown_flag.is_set():
            try:
                # Use timeout so we can check shutdown flag periodically
                try:
                    msg = self._progress_queue.get(timeout=0.1)
                except Empty:
                    continue

                task_id = msg.get("task_id")
                if not task_id:
                    continue

                msg_type = msg.get("type")

                with self._lock:
                    if task_id not in self._progress:
                        continue

                    if msg_type == "progress":
                        self._progress[task_id].downloaded_bytes = msg[
                            "downloaded_bytes"
                        ]
                        self._progress[task_id].total_bytes = msg["total_bytes"]
                        self._progress[task_id].percent = msg["percent"]
                        self._progress[task_id].speed_mbps = msg["speed_mbps"]
                        self._progress[task_id].eta_seconds = msg["eta_seconds"]
                        self._progress[task_id].status = "downloading"

                    elif msg_type == "status":
                        self._progress[task_id].status = msg["status"]

                    elif msg_type == "complete":
                        self._progress[task_id].status = "completed"
                        self._progress[task_id].percent = 100.0
                        self._progress[task_id].file_path = msg["file_path"]
                        self._progress[task_id].completed_at = time.time()
                        print(
                            f"{PRNT_API} Download completed {task_id}: {msg['file_path']}",
                            flush=True,
                        )

                        # Call on_complete callback
                        callbacks = self._callbacks.get(task_id, {})
                        on_complete = callbacks.get("on_complete")
                        if on_complete:
                            try:
                                on_complete(task_id, msg["file_path"])
                            except Exception as e:
                                print(
                                    f"{PRNT_API} Error in on_complete callback: {e}",
                                    flush=True,
                                )

                    elif msg_type == "error":
                        self._progress[task_id].status = "error"
                        self._progress[task_id].error = msg["error"]
                        self._progress[task_id].completed_at = time.time()
                        print(
                            f"{PRNT_API} Download error {task_id}: {msg['error']}",
                            flush=True,
                        )

                        # Call on_error callback
                        callbacks = self._callbacks.get(task_id, {})
                        on_error = callbacks.get("on_error")
                        if on_error:
                            try:
                                on_error(task_id, Exception(msg["error"]))
                            except Exception as e:
                                print(
                                    f"{PRNT_API} Error in on_error callback: {e}",
                                    flush=True,
                                )

            except Exception as e:
                print(f"{PRNT_API} Queue processing error: {e}", flush=True)

    def start_download(
        self,
        repo_id: str,
        filename: str,
        cache_dir: str,
        on_complete: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> str:
        """
        Start a new download task in a separate process.

        Args:
            repo_id: HuggingFace repository ID
            filename: Name of file to download
            cache_dir: Local cache directory
            on_complete: Callback(task_id, file_path) called on success
            on_error: Callback(task_id, exception) called on failure

        Returns:
            task_id: Unique identifier for tracking this download
        """
        task_id = str(uuid.uuid4())[:8]

        with self._lock:
            self._progress[task_id] = DownloadProgress(
                task_id=task_id,
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                status="pending",
            )
            self._callbacks[task_id] = {
                "on_complete": on_complete,
                "on_error": on_error,
            }

        # Create and start the download process
        process = Process(
            target=_download_worker,
            args=(task_id, repo_id, filename, cache_dir, self._progress_queue),
            daemon=True,  # Daemon process will be killed when main process exits
        )
        process.start()

        with self._lock:
            self._processes[task_id] = process

        print(
            f"{PRNT_API} Started download task {task_id}: {repo_id}/{filename}",
            flush=True,
        )
        return task_id

    def get_progress(self, task_id: str) -> Optional[dict]:
        """
        Get current progress for a download task.

        Args:
            task_id: The task identifier

        Returns:
            Progress dictionary or None if task not found.

            If this task has a linked secondary task (e.g., mmproj):
            - `status` will only be 'completed' when BOTH tasks are done
            - `primary_file_done` indicates if the primary file finished
            - `secondary_file_done` indicates if the secondary file finished
            - `secondary_progress` contains the secondary task's full progress data
        """
        with self._lock:
            if task_id not in self._progress:
                return None

            progress = self._progress[task_id].to_dict()
            primary_status = progress["status"]
            primary_done = primary_status == "completed"

            # Build primary progress object (always present)
            progress["primary_progress"] = {
                "downloaded_bytes": progress["downloaded_bytes"],
                "total_bytes": progress["total_bytes"],
                "percent": progress["percent"],
                "speed_mbps": progress["speed_mbps"],
                "eta_seconds": progress["eta_seconds"],
                "status": primary_status,
            }

            # Check if there's a linked secondary task
            secondary_id = progress.get("secondary_task_id")
            if secondary_id and secondary_id in self._progress:
                secondary_progress = self._progress[secondary_id].to_dict()
                secondary_status = secondary_progress["status"]
                secondary_done = secondary_status == "completed"

                # Add per-file completion flags
                progress["primary_file_done"] = primary_done
                progress["secondary_file_done"] = secondary_done

                # Include secondary progress data so frontend only needs one SSE stream
                progress["secondary_progress"] = {
                    "downloaded_bytes": secondary_progress["downloaded_bytes"],
                    "total_bytes": secondary_progress["total_bytes"],
                    "percent": secondary_progress["percent"],
                    "speed_mbps": secondary_progress["speed_mbps"],
                    "eta_seconds": secondary_progress["eta_seconds"],
                    "status": secondary_status,
                }

                # Override status: only 'completed' when ALL tasks are done
                # This prevents frontend from thinking download is finished prematurely
                if primary_done and not secondary_done:
                    progress["status"] = "downloading"
                elif primary_status == "error" or secondary_status == "error":
                    progress["status"] = "error"
                    if secondary_status == "error":
                        secondary_error = secondary_progress.get("error")
                        progress["error"] = (
                            f"Secondary download failed: {secondary_error}"
                        )
            else:
                # No secondary task, primary status is the overall status
                progress["primary_file_done"] = primary_done
                progress["secondary_file_done"] = None  # No secondary
                progress["secondary_progress"] = None

            return progress

    def cancel_download(self, task_id: str) -> bool:
        """
        Cancel a download task by terminating its process and cleaning up partial files.

        This ensures users can always re-download after cancellation by:
        1. Terminating the download process
        2. Removing incomplete/partial download files
        3. Removing lock files

        If this task is linked to another task (primary/secondary relationship),
        both tasks will be cancelled together.

        Args:
            task_id: The task identifier

        Returns:
            True if cancellation was successful, False if task not found
        """
        # Collect info for all tasks to cancel (may include linked tasks)
        tasks_to_cancel = []

        with self._lock:
            if task_id not in self._processes:
                return False

            # Capture cleanup info while holding the lock. We extract all needed
            # data (cache_dir, repo_id, filename) into local variables here so
            # that file cleanup can safely happen outside the lock. This avoids
            # blocking other operations during potentially slow I/O, and the
            # captured values remain valid even if _progress is modified later.
            progress = self._progress.get(task_id)
            if progress:
                tasks_to_cancel.append({
                    "task_id": task_id,
                    "process": self._processes.get(task_id),
                    "cache_dir": progress.cache_dir,
                    "repo_id": progress.repo_id,
                    "filename": progress.filename,
                })

                # Check for linked secondary task (e.g., mmproj file)
                secondary_id = progress.secondary_task_id
                if secondary_id and secondary_id in self._progress:
                    secondary_progress = self._progress[secondary_id]
                    tasks_to_cancel.append({
                        "task_id": secondary_id,
                        "process": self._processes.get(secondary_id),
                        "cache_dir": secondary_progress.cache_dir,
                        "repo_id": secondary_progress.repo_id,
                        "filename": secondary_progress.filename,
                    })

            # Also check if this task IS a secondary task of some primary
            # (i.e., some other task has this task_id as its secondary_task_id)
            for other_task_id, other_progress in self._progress.items():
                if other_task_id == task_id:
                    continue
                if other_progress.secondary_task_id == task_id:
                    # This task is a secondary, so cancel its primary too
                    already_included = any(
                        t["task_id"] == other_task_id for t in tasks_to_cancel
                    )
                    if not already_included:
                        tasks_to_cancel.append({
                            "task_id": other_task_id,
                            "process": self._processes.get(other_task_id),
                            "cache_dir": other_progress.cache_dir,
                            "repo_id": other_progress.repo_id,
                            "filename": other_progress.filename,
                        })
                    break  # There should only be one primary per secondary

            # Terminate all processes and update their status
            for task_info in tasks_to_cancel:
                tid = task_info["task_id"]
                process = task_info["process"]

                if process and process.is_alive():
                    print(
                        f"{PRNT_API} Terminating download process {tid}", flush=True
                    )
                    process.terminate()
                    process.join(timeout=1.0)
                    if process.is_alive():
                        print(
                            f"{PRNT_API} Force killing download process {tid}",
                            flush=True,
                        )
                        process.kill()
                        process.join(timeout=1.0)

                # Update status
                if tid in self._progress:
                    self._progress[tid].status = "cancelled"
                    self._progress[tid].completed_at = time.time()

                print(f"{PRNT_API} Download cancelled {tid}", flush=True)

        # Clean up incomplete files OUTSIDE the lock to avoid blocking other
        # operations. This removes .incomplete and .lock files so user can
        # re-download. Safe to use captured values here since they're immutable
        # copies taken while we held the lock.
        for task_info in tasks_to_cancel:
            cache_dir = task_info["cache_dir"]
            repo_id = task_info["repo_id"]
            filename = task_info["filename"]
            if cache_dir and repo_id:
                _cleanup_incomplete_download(cache_dir, repo_id, filename)

        return True

    def get_all_downloads(self) -> Dict[str, dict]:
        """Get progress for all active downloads."""
        # Run cleanup of stale tasks first
        self._cleanup_stale_tasks()

        with self._lock:
            return {
                task_id: progress.to_dict()
                for task_id, progress in self._progress.items()
                if progress.status in ("pending", "downloading")
            }

    def set_secondary_task(self, primary_task_id: str, secondary_task_id: str) -> bool:
        """
        Link a secondary download task to a primary task.
        Used for chained downloads like mmproj files.

        Args:
            primary_task_id: The main download task ID
            secondary_task_id: The secondary download task ID

        Returns:
            True if successful, False if primary task not found
        """
        with self._lock:
            if primary_task_id in self._progress:
                self._progress[primary_task_id].secondary_task_id = secondary_task_id
                return True
        return False

    def cleanup_task(self, task_id: str) -> None:
        """Remove a completed/cancelled/errored task from tracking."""
        with self._lock:
            self._progress.pop(task_id, None)
            self._callbacks.pop(task_id, None)

            # Clean up process reference
            if task_id in self._processes:
                process = self._processes.pop(task_id)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=0.5)

    def _cleanup_stale_tasks(self) -> None:
        """Remove tasks that have been completed for longer than TASK_RETENTION_SECONDS."""
        current_time = time.time()
        tasks_to_remove = []

        with self._lock:
            for task_id, progress in self._progress.items():
                if progress.completed_at is not None:
                    age = current_time - progress.completed_at
                    if age > self.TASK_RETENTION_SECONDS:
                        tasks_to_remove.append(task_id)

        # Remove outside of the iteration to avoid modifying dict during iteration
        for task_id in tasks_to_remove:
            self.cleanup_task(task_id)
            print(f"{PRNT_API} Auto-cleaned stale task {task_id}", flush=True)

    def shutdown(self) -> None:
        """
        Shutdown the download manager and terminate all downloads immediately.
        """
        print(f"{PRNT_API} Download manager shutting down...", flush=True)

        # Set shutdown flag to stop queue processing thread
        self._shutdown_flag.set()

        # Terminate all running download processes
        with self._lock:
            for task_id, process in list(self._processes.items()):
                if process.is_alive():
                    print(f"{PRNT_API} Terminating download {task_id}", flush=True)
                    process.terminate()
                    process.join(timeout=0.5)
                    if process.is_alive():
                        process.kill()

            self._processes.clear()

        print(f"{PRNT_API} Download manager shutdown complete", flush=True)


# Global instance (initialized in api_server.py)
download_manager: Optional[DownloadManager] = None


def get_download_manager() -> DownloadManager:
    """Get the global download manager instance."""
    global download_manager
    if download_manager is None:
        download_manager = DownloadManager(max_workers=3)
    return download_manager


def init_download_manager(max_workers: int = 3) -> DownloadManager:
    """Initialize the global download manager."""
    global download_manager
    download_manager = DownloadManager(max_workers=max_workers)
    return download_manager
