import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request
from core import classes, common
from .harness import AgenticSearch, SearchResult
from .classes import (
    FileSystemSearchRequest,
    VectorSearchRequest,
    WebSearchRequest,
    StructuredSearchRequest,
)
from .providers import (
    FileSystemProvider,
    VectorProvider,
    WebProvider,
    StructuredProvider,
)


router = APIRouter()


def _get_active_searches(app: classes.FastAPIApp) -> dict:
    """Get or initialize the active_searches dict on app state."""
    if not hasattr(app.state, "active_searches"):
        app.state.active_searches = {}
    return app.state.active_searches


@router.post("/stop")
async def stop_search(request: Request, search_id: Optional[str] = None):
    """
    Stop an active search by ID, or all active searches if no ID provided.

    Args:
        search_id: Optional specific search ID to stop. If not provided, stops all.

    Returns:
        Success status and message.
    """
    app: classes.FastAPIApp = request.app
    active_searches = _get_active_searches(app)

    if search_id:
        if search_id in active_searches:
            active_searches[search_id].abort_requested = True
            print(
                f"{common.PRNT_API} Stop requested for search {search_id}", flush=True
            )
            return {
                "success": True,
                "message": f"Stop requested for search {search_id}.",
            }
        return {"success": False, "message": f"Search {search_id} not found."}
    else:
        # Stop all active searches
        count = len(active_searches)
        for orchestrator in active_searches.values():
            orchestrator.abort_requested = True
        print(
            f"{common.PRNT_API} Stop requested for {count} active searches", flush=True
        )
        return {
            "success": True,
            "message": f"Stop requested for {count} active searches.",
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

    If no collection is specified, operates in discovery mode - listing all
    available collections and letting the LLM select which to search.

    Requires:
    - A loaded LLM model
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

        # Create provider with optional collections list
        provider = VectorProvider(
            app=app,
            collections=payload.collections,  # Can be None for discovery mode
            top_k=payload.top_k or 50,
        )

        search_id = str(uuid.uuid4())
        active_searches = _get_active_searches(app)

        try:
            orchestrator = AgenticSearch(
                provider=provider,
                llm=app.state.llm,
                search_type="vector",
            )
            orchestrator.search_id = search_id
            active_searches[search_id] = orchestrator

            # Run the search - initial_scope not used, collections passed via provider
            result = await orchestrator.search(
                query=payload.query,
                initial_scope=None,  # VectorProvider uses self.collections instead
                max_preview=payload.max_preview or 10,
                max_extract=payload.max_extract or 3,
                auto_expand=(
                    payload.auto_expand if payload.auto_expand is not None else True
                ),
                request=request,
            )

            return result

        finally:
            # Clean up orchestrator from active searches
            active_searches.pop(search_id, None)
            # Clean up vision embedder if it was used
            await provider.close()

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
            website=payload.website,
            max_pages=payload.max_pages or 10,
        )

        search_id = str(uuid.uuid4())
        active_searches = _get_active_searches(app)

        try:
            orchestrator = AgenticSearch(
                provider=provider,
                llm=app.state.llm,
                search_type="web",
            )
            orchestrator.search_id = search_id
            active_searches[search_id] = orchestrator

            # Run the search
            result = await orchestrator.search(
                query=payload.query,
                initial_scope=payload.query,  # For web search, scope is the query itself
                max_preview=payload.max_preview or 10,
                max_extract=payload.max_extract or 3,
                auto_expand=False,  # Web search doesn't use scope expansion
                request=request,
            )

            return result

        finally:
            # Clean up orchestrator from active searches
            active_searches.pop(search_id, None)
            # Clean up HTTP client (runs even if exception occurs)
            await provider.close()

    except Exception as e:
        print(f"{common.PRNT_API} Web search error: {e}", flush=True)
        return SearchResult(
            success=False,
            message=f"Web search failed: {e}",
            data=None,
        )


# File system search endpoint using AgenticSearch
@router.post("/fs")
async def search_file_system(
    request: Request,
    payload: FileSystemSearchRequest,
) -> SearchResult:
    """
    Perform agentic file search with the unified multi-phase strategy.

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

        search_id = str(uuid.uuid4())
        active_searches = _get_active_searches(app)

        try:
            orchestrator = AgenticSearch(
                provider=provider,
                llm=app.state.llm,
                search_type="filesystem",
            )
            orchestrator.search_id = search_id
            active_searches[search_id] = orchestrator

            # Run the search
            result = await orchestrator.search(
                query=payload.query,
                initial_scope=payload.directory,
                max_preview=payload.max_files_preview or 10,
                max_extract=payload.max_files_parse or 3,
                auto_expand=(
                    payload.auto_expand if payload.auto_expand is not None else True
                ),
                request=request,
            )

            return result

        finally:
            # Clean up orchestrator from active searches
            active_searches.pop(search_id, None)

    except Exception as e:
        print(f"{common.PRNT_API} File search error: {e}", flush=True)
        return SearchResult(
            success=False,
            message=f"File search failed: {e}",
            data=None,
        )


# Structured data search using AgenticSearch with StructuredProvider
@router.post("/structured")
async def search_structured(
    request: Request,
    payload: StructuredSearchRequest,
) -> SearchResult:
    """
    Perform agentic search over client-provided structured data.

    The agent searches over ephemeral data sent in the request,
    selects relevant items, and synthesizes an answer using the LLM.

    Use cases:
    - Searching conversation history
    - Finding relevant project metadata
    - Searching workflow data
    - Any data the frontend has that the server cannot access

    Requires:
    - A loaded LLM model
    - At least one item in the items array
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

        # Validate items exist
        if not payload.items:
            return SearchResult(
                success=False,
                message="No items provided. Include at least one item to search.",
                data=None,
            )

        # Create provider with the provided items
        provider = StructuredProvider(
            app=app,
            items=payload.items,
            group_by=payload.group_by,
        )

        # Determine initial scope from first group if grouping enabled
        initial_scope = None
        if payload.group_by and provider._groups:
            initial_scope = list(provider._groups.keys())[0]

        search_id = str(uuid.uuid4())
        active_searches = _get_active_searches(app)

        try:
            orchestrator = AgenticSearch(
                provider=provider,
                llm=app.state.llm,
                search_type="structured",
            )
            orchestrator.search_id = search_id
            active_searches[search_id] = orchestrator

            # Run the search
            result = await orchestrator.search(
                query=payload.query,
                initial_scope=initial_scope,
                max_preview=payload.max_preview or 10,
                max_extract=payload.max_extract or 3,
                auto_expand=payload.auto_expand if payload.auto_expand is not None else False,
                request=request,
            )

            return result

        finally:
            # Clean up orchestrator from active searches
            active_searches.pop(search_id, None)

    except Exception as e:
        print(f"{common.PRNT_API} Structured search error: {e}", flush=True)
        return SearchResult(
            success=False,
            message=f"Structured search failed: {e}",
            data=None,
        )
