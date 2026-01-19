from pathlib import Path
from fastapi import APIRouter, Request
from core import classes, common
from .search_fs import SearchFS
from .base import AgenticSearch, SearchResult
from .classes import (
    FileSystemSearchRequest,
    FileSystemSearchResponse,
    VectorSearchRequest,
    WebSearchRequest,
)
from .providers import FileSystemProvider, VectorProvider, WebProvider


router = APIRouter()


# Structured file system search (scan, preview, deep dive)
@router.post("/fs")
async def search_file_system(
    request: Request,
    payload: FileSystemSearchRequest,
) -> FileSystemSearchResponse:
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


# Vector/Embedding search using AgenticSearch with VectorProvider
@router.post("/vector")
async def search_vector(
    request: Request,
    payload: VectorSearchRequest,
) -> SearchResult:
    """
    Perform agentic vector/embedding search with multi-phase strategy.

    The agent queries ChromaDB collections, selects relevant chunks,
    expands context, and synthesizes an answer using the LLM.

    Requires:
    - A loaded LLM model
    - Valid collection name in allowed_collections
    """
    app: classes.FastAPIApp = request.app

    try:
        # Verify LLM is loaded
        if not app.state.llm:
            return SearchResult(
                success=False,
                message="No LLM loaded. Load a model first.",
                data=None,
            )

        # Validate that collection is in allowed list
        if payload.collection not in payload.allowed_collections:
            return SearchResult(
                success=False,
                message=f"Collection '{payload.collection}' is not in allowed collections.",
                data=None,
            )

        # Create provider and orchestrator
        provider = VectorProvider(
            app=app,
            allowed_collections=payload.allowed_collections,
            top_k=payload.top_k or 50,
        )

        orchestrator = AgenticSearch(
            provider=provider,
            llm=app.state.llm,
            search_type="vector",
        )

        # Run the search
        result = await orchestrator.search(
            query=payload.query,
            initial_scope=payload.collection,
            max_preview=payload.max_preview or 10,
            max_extract=payload.max_extract or 3,
            auto_expand=payload.auto_expand if payload.auto_expand is not None else True,
        )

        return result

    except Exception as e:
        print(f"{common.PRNT_API} Vector search error: {e}", flush=True)
        return SearchResult(
            success=False,
            message=f"Vector search failed: {e}",
            data=None,
        )


# Web search using AgenticSearch with WebProvider (DuckDuckGo)
@router.post("/web")
async def search_web(
    request: Request,
    payload: WebSearchRequest,
) -> SearchResult:
    """
    Perform agentic web search with multi-phase strategy using DuckDuckGo.

    The agent searches the web, selects relevant URLs, fetches page content,
    and synthesizes an answer using the LLM.

    Requires:
    - A loaded LLM model
    - Optional: website or allowed_domains for filtering
    """
    app: classes.FastAPIApp = request.app

    try:
        # Verify LLM is loaded
        if not app.state.llm:
            return SearchResult(
                success=False,
                message="No LLM loaded. Load a model first.",
                data=None,
            )

        # Create provider and orchestrator
        provider = WebProvider(
            app=app,
            allowed_domains=payload.allowed_domains,
            website=payload.website,
            max_pages=payload.max_pages or 10,
        )

        try:
            orchestrator = AgenticSearch(
                provider=provider,
                llm=app.state.llm,
                search_type="web",
            )

            # Run the search
            result = await orchestrator.search(
                query=payload.query,
                initial_scope=payload.query,  # For web search, scope is the query itself
                max_preview=payload.max_preview or 10,
                max_extract=payload.max_extract or 3,
                auto_expand=False,  # Web search doesn't use scope expansion
            )

            return result

        finally:
            # Clean up HTTP client (runs even if exception occurs)
            await provider.close()

    except Exception as e:
        print(f"{common.PRNT_API} Web search error: {e}", flush=True)
        return SearchResult(
            success=False,
            message=f"Web search failed: {e}",
            data=None,
        )


# New unified file system search endpoint using AgenticSearch
@router.post("/fs/v2")
async def search_file_system_v2(
    request: Request,
    payload: FileSystemSearchRequest,
) -> SearchResult:
    """
    Perform agentic file search with the unified multi-phase strategy.

    This is the new version using the AgenticSearch orchestrator
    with FileSystemProvider. The original /fs endpoint is preserved
    for backwards compatibility.

    Requires:
    - A loaded LLM model
    - allowed_directories must include the search directory
    """
    app: classes.FastAPIApp = request.app

    try:
        # Verify LLM is loaded
        if not app.state.llm:
            return SearchResult(
                success=False,
                message="No LLM loaded. Load a model first.",
                data=None,
            )

        # Validate that search directory is in allowed list
        if payload.directory not in payload.allowed_directories:
            search_path = Path(payload.directory).resolve()
            is_allowed = any(
                search_path == Path(d).resolve()
                or search_path.is_relative_to(Path(d).resolve())
                for d in payload.allowed_directories
            )
            if not is_allowed:
                return SearchResult(
                    success=False,
                    message=f"Search directory '{payload.directory}' is not in allowed directories.",
                    data=None,
                )

        # Create provider and orchestrator
        provider = FileSystemProvider(
            app=app,
            allowed_directories=payload.allowed_directories,
            file_patterns=payload.file_patterns,
        )

        orchestrator = AgenticSearch(
            provider=provider,
            llm=app.state.llm,
            search_type="filesystem",
        )

        # Run the search
        result = await orchestrator.search(
            query=payload.query,
            initial_scope=payload.directory,
            max_preview=payload.max_files_preview or 10,
            max_extract=payload.max_files_parse or 3,
            auto_expand=payload.auto_expand if payload.auto_expand is not None else True,
        )

        return result

    except Exception as e:
        print(f"{common.PRNT_API} File search v2 error: {e}", flush=True)
        return SearchResult(
            success=False,
            message=f"File search failed: {e}",
            data=None,
        )
