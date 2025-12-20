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
    """
    app: classes.FastAPIApp = request.app
    download_manager = app.state.download_manager

    async def event_generator():
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    print(
                        f"{common.PRNT_API} SSE client disconnected for task {task_id}",
                        flush=True,
                    )
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
                    break

                # Poll interval - balance between responsiveness and CPU usage
                await asyncio.sleep(0.3)

        except asyncio.CancelledError:
            print(
                f"{common.PRNT_API} SSE stream cancelled for task {task_id}", flush=True
            )
        except Exception as e:
            print(f"{common.PRNT_API} SSE error for task {task_id}: {e}", flush=True)
            yield {"event": "error", "data": json.dumps({"error": str(e)})}

    return EventSourceResponse(event_generator())


@router.delete("")
def cancel_download(task_id: str, request: Request):
    """
    Cancel an in-progress download.

    Works for any model type since task_id is globally unique.
    """
    app: classes.FastAPIApp = request.app
    download_manager = app.state.download_manager

    success = download_manager.cancel_download(task_id)

    if success:
        return {
            "success": True,
            "message": f"Cancellation requested for task {task_id}",
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
