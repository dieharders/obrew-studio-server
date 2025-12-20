"""
Download Manager for HuggingFace Hub downloads with progress tracking.

Provides non-blocking concurrent downloads using ThreadPoolExecutor,
with real-time progress updates accessible via SSE endpoints.
"""

import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Callable, Any
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from core.common import PRNT_API


@dataclass
class DownloadProgress:
    """Represents the current state of a download task."""

    task_id: str
    repo_id: str
    filename: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    percent: float = 0.0
    speed_mbps: float = 0.0
    eta_seconds: Optional[int] = None
    status: str = "pending"  # pending, downloading, completed, error, cancelled
    error: Optional[str] = None
    file_path: Optional[str] = None  # Set on completion

    def to_dict(self) -> dict:
        return asdict(self)


class ProgressCaptureTqdm(tqdm):
    """
    Custom tqdm class that captures progress and reports via callback.
    Used to intercept huggingface_hub download progress.
    """

    def __init__(self, *args, progress_callback: Callable[[dict], None] = None, **kwargs):
        self._progress_callback = progress_callback
        self._last_update_time = time.time()
        self._update_interval = 0.2  # Minimum seconds between updates
        # Filter out huggingface-hub v1.x XET-specific arguments that tqdm doesn't recognize
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

    def update(self, n: int = 1) -> bool | None:
        result = super().update(n)

        # Throttle updates to avoid overwhelming the system
        current_time = time.time()
        if current_time - self._last_update_time < self._update_interval:
            return result

        self._last_update_time = current_time

        if self._progress_callback and self.total:
            # Calculate speed in MB/s
            rate = self.format_dict.get("rate", 0) or 0
            speed_mbps = rate / (1024 * 1024) if rate else 0.0

            # Calculate ETA
            eta = None
            if rate and rate > 0:
                remaining = self.total - self.n
                eta = int(remaining / rate)

            self._progress_callback(
                {
                    "downloaded_bytes": self.n,
                    "total_bytes": self.total,
                    "percent": (self.n / self.total * 100) if self.total else 0,
                    "speed_mbps": round(speed_mbps, 2),
                    "eta_seconds": eta,
                }
            )
        return result


class DownloadManager:
    """
    Manages concurrent HuggingFace model downloads with progress tracking.

    Features:
    - Non-blocking downloads using ThreadPoolExecutor
    - Real-time progress tracking
    - Support for cancellation
    - Thread-safe progress state
    """

    def __init__(self, max_workers: int = 3):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._progress: Dict[str, DownloadProgress] = {}
        self._futures: Dict[str, Future] = {}
        self._cancel_flags: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def start_download(
        self,
        repo_id: str,
        filename: str,
        cache_dir: str,
        on_complete: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
    ) -> str:
        """
        Start a new download task.

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
                status="pending",
            )
            self._cancel_flags[task_id] = threading.Event()

        future = self._executor.submit(
            self._download_task,
            task_id,
            repo_id,
            filename,
            cache_dir,
            on_complete,
            on_error,
        )

        with self._lock:
            self._futures[task_id] = future

        print(f"{PRNT_API} Started download task {task_id}: {repo_id}/{filename}", flush=True)
        return task_id

    def _download_task(
        self,
        task_id: str,
        repo_id: str,
        filename: str,
        cache_dir: str,
        on_complete: Optional[Callable],
        on_error: Optional[Callable],
    ) -> None:
        """Internal download task executed in thread pool."""

        def progress_callback(progress_data: dict) -> None:
            # Check for cancellation
            if self._cancel_flags.get(task_id, threading.Event()).is_set():
                raise InterruptedError("Download cancelled")

            with self._lock:
                if task_id in self._progress:
                    self._progress[task_id].downloaded_bytes = progress_data["downloaded_bytes"]
                    self._progress[task_id].total_bytes = progress_data["total_bytes"]
                    self._progress[task_id].percent = progress_data["percent"]
                    self._progress[task_id].speed_mbps = progress_data["speed_mbps"]
                    self._progress[task_id].eta_seconds = progress_data["eta_seconds"]
                    self._progress[task_id].status = "downloading"

        # Create factory for tqdm class with bound callback
        def make_tqdm_class():
            class BoundProgressTqdm(ProgressCaptureTqdm):
                def __init__(self, *args, **kwargs):
                    kwargs["progress_callback"] = progress_callback
                    super().__init__(*args, **kwargs)

            return BoundProgressTqdm

        try:
            with self._lock:
                self._progress[task_id].status = "downloading"

            # Perform the download with progress tracking
            # Note: resume_download was removed in huggingface-hub v1.0 (now automatic)
            file_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                tqdm_class=make_tqdm_class(),
            )

            # Check if cancelled during download
            if self._cancel_flags.get(task_id, threading.Event()).is_set():
                with self._lock:
                    self._progress[task_id].status = "cancelled"
                return

            # Mark as completed
            with self._lock:
                self._progress[task_id].status = "completed"
                self._progress[task_id].percent = 100.0
                self._progress[task_id].file_path = file_path

            print(f"{PRNT_API} Download completed {task_id}: {file_path}", flush=True)

            if on_complete:
                on_complete(task_id, file_path)

        except InterruptedError:
            with self._lock:
                self._progress[task_id].status = "cancelled"
            print(f"{PRNT_API} Download cancelled {task_id}", flush=True)

        except Exception as e:
            with self._lock:
                self._progress[task_id].status = "error"
                self._progress[task_id].error = str(e)

            print(f"{PRNT_API} Download error {task_id}: {e}", flush=True)

            if on_error:
                on_error(task_id, e)

    def get_progress(self, task_id: str) -> Optional[dict]:
        """
        Get current progress for a download task.

        Args:
            task_id: The task identifier

        Returns:
            Progress dictionary or None if task not found
        """
        with self._lock:
            if task_id in self._progress:
                return self._progress[task_id].to_dict()
        return None

    def cancel_download(self, task_id: str) -> bool:
        """
        Request cancellation of a download task.

        Args:
            task_id: The task identifier

        Returns:
            True if cancellation was requested, False if task not found
        """
        with self._lock:
            if task_id in self._cancel_flags:
                self._cancel_flags[task_id].set()
                print(f"{PRNT_API} Cancellation requested for {task_id}", flush=True)
                return True
        return False

    def get_all_downloads(self) -> Dict[str, dict]:
        """Get progress for all active downloads."""
        with self._lock:
            return {
                task_id: progress.to_dict()
                for task_id, progress in self._progress.items()
                if progress.status in ("pending", "downloading")
            }

    def cleanup_task(self, task_id: str) -> None:
        """Remove a completed/cancelled/errored task from tracking."""
        with self._lock:
            self._progress.pop(task_id, None)
            self._futures.pop(task_id, None)
            self._cancel_flags.pop(task_id, None)

    def shutdown(self) -> None:
        """Shutdown the executor and cancel all pending downloads."""
        # Signal all downloads to cancel
        with self._lock:
            for cancel_flag in self._cancel_flags.values():
                cancel_flag.set()

        self._executor.shutdown(wait=False)
        print(f"{PRNT_API} Download manager shutdown", flush=True)


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
