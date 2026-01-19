from pathlib import Path
from fastapi import APIRouter, Request
from core import classes, common
from .search_fs import SearchFS
from .classes import FileSearchRequest, FileSearchResponse


router = APIRouter()


# Structured file system search (scan, preview, deep dive)
@router.post("/fs")
async def file_search(
    request: Request,
    payload: FileSearchRequest,
) -> FileSearchResponse:
    """
    Perform agentic file search with multi-phase strategy.

    The agent scans directories, previews files, parses relevant documents,
    and synthesizes an answer using the LLM.

    Requires:
    - A loaded LLM model
    - allowed_directories must include the search directory
    """
    app: classes.FastAPIApp = request.app

    try:
        # Verify LLM is loaded
        if not app.state.llm:
            return {
                "success": False,
                "message": "No LLM loaded. Load a model first.",
                "data": None,
            }

        # Validate that search directory is in allowed list
        if payload.directory not in payload.allowed_directories:
            # Check if it's a subdirectory of an allowed directory
            search_path = Path(payload.directory).resolve()
            is_allowed = any(
                search_path == Path(d).resolve()
                or search_path.is_relative_to(Path(d).resolve())
                for d in payload.allowed_directories
            )
            if not is_allowed:
                return {
                    "success": False,
                    "message": f"Search directory '{payload.directory}' is not in allowed directories.",
                    "data": None,
                }

        # Create and run search agent
        agent = SearchFS(
            app=app,
            allowed_directories=payload.allowed_directories,
        )

        result = await agent.search(
            query=payload.query,
            directory=payload.directory,
            max_files_preview=payload.max_files_preview or 10,
            max_files_parse=payload.max_files_parse or 3,
            file_patterns=payload.file_patterns,
        )

        return {
            "success": True,
            "message": "Search completed.",
            "data": result,
        }

    except Exception as e:
        print(f"{common.PRNT_API} File search error: {e}", flush=True)
        return {
            "success": False,
            "message": f"File search failed: {e}",
            "data": None,
        }
