from pathlib import Path
from fastapi import APIRouter, Request
from core import classes, common
from .search_fs import SearchFS
from .agentic_search import AgenticSearchAgent
from .classes import (
    FileSearchRequest,
    FileSearchResponse,
    AgenticSearchRequest,
    AgenticSearchResponse,
)


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


# True agentic file search (LLM chooses tools)
@router.post("/agentic-fs")
async def agentic_file_search(
    request: Request,
    payload: AgenticSearchRequest,
) -> AgenticSearchResponse:
    """
    Perform true agentic file search where the LLM decides which tools to use.

    Unlike /search which follows a fixed flow, this endpoint lets the LLM:
    - See all available file tools and their descriptions
    - Decide which tool to call at each step
    - Loop until it has enough information to answer

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

        # Create and run agentic search agent
        agent = AgenticSearchAgent(
            app=app,
            allowed_directories=payload.allowed_directories,
        )

        result = await agent.search(
            query=payload.query,
            directory=payload.directory,
            max_iterations=payload.max_iterations or 10,
            file_patterns=payload.file_patterns,
        )

        return {
            "success": True,
            "message": f"Agentic search completed in {result.get('iterations', 0)} iterations.",
            "data": result,
        }

    except Exception as e:
        print(f"{common.PRNT_API} Agentic search error: {e}", flush=True)
        return {
            "success": False,
            "message": f"Agentic search failed: {e}",
            "data": None,
        }
