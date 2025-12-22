"""
Centralized download progress tracking and management endpoints.

Provides a unified API for tracking download progress across all model types
(text, embedding, vision) without requiring model-type-specific endpoints.
"""

import json
import asyncio
from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from core import classes, common

router = APIRouter()


@router.get("/progress")
async def download_progress(task_id: str, request: Request):
    """
    Stream download progress via SSE for any model type.

    Returns real-time progress updates including:
    - downloaded_bytes / total_bytes
    - percent complete
    - speed_mbps
    - eta_seconds
    - status (pending, downloading, completed, error, cancelled)
    - secondary_task_id (if a chained download like mmproj was started)

    After the stream ends (completed/error/cancelled), the task is automatically
    cleaned up from memory.
    """
    app: classes.FastAPIApp = request.app
    download_manager = app.state.download_manager

    async def event_generator():
        should_cleanup = False
        client_disconnected = False
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    print(
                        f"{common.PRNT_API} SSE client disconnected for task {task_id}",
                        flush=True,
                    )
                    client_disconnected = True
                    break

                progress = download_manager.get_progress(task_id)

                if progress is None:
                    # Task not found - send error and close
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": f"Task {task_id} not found"}),
                    }
                    break

                # Send progress update
                yield {"event": "progress", "data": json.dumps(progress)}

                # Check if download is complete or failed
                if progress.get("status") in ["completed", "error", "cancelled"]:
                    # Send final event and close stream
                    yield {"event": "done", "data": json.dumps(progress)}
                    should_cleanup = True
                    break

                # Poll interval - balance between responsiveness and CPU usage
                await asyncio.sleep(0.3)

        except asyncio.CancelledError:
            # SSE stream was aborted by client - cancel the underlying download
            print(
                f"{common.PRNT_API} SSE stream cancelled for task {task_id}, cancelling download",
                flush=True,
            )
            # Also cancel secondary task (e.g., mmproj) if present
            progress = download_manager.get_progress(task_id)
            if progress and progress.get("secondary_task_id"):
                secondary_id = progress["secondary_task_id"]
                download_manager.cancel_download(secondary_id)
                print(
                    f"{common.PRNT_API} Also cancelled secondary task {secondary_id}",
                    flush=True,
                )
            download_manager.cancel_download(task_id)
            # Don't cleanup immediately - let the download thread finish and set cancelled status
            # The stale task cleanup will remove it eventually
        except Exception as e:
            print(f"{common.PRNT_API} SSE error for task {task_id}: {e}", flush=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}
        finally:
            # If client disconnected (closed browser, navigated away, etc.), cancel the download
            if client_disconnected:
                print(
                    f"{common.PRNT_API} Client disconnected, cancelling download for task {task_id}",
                    flush=True,
                )
                # Cancel secondary task (e.g., mmproj) if present
                progress = download_manager.get_progress(task_id)
                if progress and progress.get("secondary_task_id"):
                    secondary_id = progress["secondary_task_id"]
                    download_manager.cancel_download(secondary_id)
                    print(
                        f"{common.PRNT_API} Also cancelled secondary task {secondary_id}",
                        flush=True,
                    )
                download_manager.cancel_download(task_id)
                # Don't cleanup - let download thread finish and set cancelled status

            # Cleanup task after stream ends with terminal status (client has received final status)
            # Don't cleanup on SSE abort or disconnect - the download needs the cancel flag to stop
            if should_cleanup:
                download_manager.cleanup_task(task_id)

    return EventSourceResponse(event_generator())


@router.delete("")
def cancel_download(task_id: str, request: Request):
    """
    Cancel an in-progress download.

    Works for any model type since task_id is globally unique.
    Also cancels any secondary task (e.g., mmproj) linked to this download.
    """
    app: classes.FastAPIApp = request.app
    download_manager = app.state.download_manager

    # Also cancel secondary task (e.g., mmproj) if present
    progress = download_manager.get_progress(task_id)
    secondary_cancelled = False
    if progress and progress.get("secondary_task_id"):
        secondary_id = progress["secondary_task_id"]
        secondary_cancelled = download_manager.cancel_download(secondary_id)
        if secondary_cancelled:
            print(
                f"{common.PRNT_API} Also cancelled secondary task {secondary_id}",
                flush=True,
            )

    success = download_manager.cancel_download(task_id)

    if success:
        message = f"Cancellation requested for task {task_id}"
        if secondary_cancelled:
            message += f" and secondary task {progress['secondary_task_id']}"
        return {
            "success": True,
            "message": message,
        }
    else:
        return {
            "success": False,
            "message": f"Task {task_id} not found or already completed",
        }


@router.get("")
def get_all_downloads(request: Request):
    """
    Get status of all active downloads.

    Returns a dictionary of task_id -> progress for all pending/downloading tasks.
    """
    app: classes.FastAPIApp = request.app
    download_manager = app.state.download_manager

    downloads = download_manager.get_all_downloads()

    return {
        "success": True,
        "message": f"Found {len(downloads)} active download(s)",
        "data": downloads,
    }
