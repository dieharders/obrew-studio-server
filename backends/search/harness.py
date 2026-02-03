"""
Base classes for the search harness.

This module provides the core abstractions for a generalized multi-phase search loop
that works across different context types (file system, vector/embedding, web).

The search loop follows these phases:
1. DISCOVER - Find items in scope (directory, collection, search query)
2. LLM SELECT - Use LLM to select relevant items for preview
3. PREVIEW - Get preview content for selected items
4. LLM DEEP SELECT - Use LLM to select items for full extraction
5. EXTRACT - Get full content for selected items
6. EXPAND - Search additional scopes if needed
7. SYNTHESIZE - Generate final answer from context
"""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Protocol, AsyncIterator, runtime_checkable
from pydantic import BaseModel
from core import common
from inference.helpers import read_event_data

# =============================================================================
# Default limits (used as defaults for configurable parameters)
# =============================================================================
DEFAULT_MAX_PREVIEW = 10  # Max items to preview in LLM selection
DEFAULT_MAX_READ = 3  # Max items to read/extract full content from
DEFAULT_MAX_EXPAND = 2  # Max additional scopes to search during expansion
DEFAULT_MAX_DISCOVER_ITEMS = 50  # Max items to return from discover phase
DEFAULT_CONTENT_PREVIEW_LENGTH = 100  # Preview length in LLM selection prompt
DEFAULT_CONTENT_SNIPPET_LENGTH = 200  # Snippet length for source citations
DEFAULT_CONTENT_EXTRACT_LENGTH = 5000  # Max content length per extracted item


# =============================================================================
# LLM Protocol - defines the interface expected by AgenticSearch
# =============================================================================
@runtime_checkable
class LLMProtocol(Protocol):
    """Protocol defining the LLM interface required by AgenticSearch."""

    async def text_completion(
        self,
        prompt: str,
        system_message: str,
        stream: bool,
        request: Any,
        constrain_json_output: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[Any]:
        """Generate text completion from the LLM."""
        ...


# JSON Schema for constrained selection output - just an array of indices
SELECTION_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "Array of item indices to select",
}


class SearchItem(BaseModel):
    """Universal item format for search results during the search loop."""

    id: str
    name: str
    type: str  # "file", "chunk", "url"
    preview: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    requires_extraction: bool = False  # Whether full content extraction is needed


class SearchSource(BaseModel):
    """Citation format for final answer."""

    id: str
    type: str
    name: str
    snippet: Optional[str] = None


class SearchResultData(BaseModel):
    """Data payload for search results."""

    answer: str
    sources: List[SearchSource]
    query: str
    search_type: str  # "filesystem" | "vector" | "web"
    total_results: int = 0  # Number of sources used to generate the answer
    stats: Optional[Dict[str, Any]] = None
    tool_logs: Optional[List[Dict[str, Any]]] = None


class SearchResult(BaseModel):
    """Unified search response format for all endpoints."""

    success: bool
    message: str
    data: Optional[SearchResultData] = None


class SearchProvider(ABC):
    """
    Protocol for context-specific search implementations.

    Each provider implements the core methods for a specific context type
    (file system, vector/embedding, web). The AgenticSearch orchestrator
    calls these methods in sequence to perform the multi-phase search.
    """

    @abstractmethod
    async def discover(self, scope: str, **kwargs) -> List[SearchItem]:
        """
        Discover items in the given scope.

        Args:
            scope: The scope to search (directory path, collection name, search query)
            **kwargs: Additional arguments (e.g., query for semantic search, file_patterns)

        Returns:
            List of SearchItem objects representing discovered items
        """
        pass

    @abstractmethod
    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given items.

        Args:
            items: List of items to preview

        Returns:
            Same items with preview field populated
        """
        pass

    @abstractmethod
    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given items.

        Args:
            items: List of items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        pass

    @abstractmethod
    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return additional scopes that can be searched if needed.

        Args:
            current_scope: The scope that was just searched

        Returns:
            List of additional scope strings to search
        """
        pass


class AgenticSearch:
    """
    Orchestrates the multi-phase search loop for any SearchProvider.

    This class implements the generalized agentic search pattern that works
    across different context types. It uses LLM-based selection at multiple
    stages to progressively narrow down the search results.
    """

    def __init__(
        self,
        provider: SearchProvider,
        llm: LLMProtocol,
        search_type: str,
    ):
        """
        Initialize the AgenticSearch orchestrator.

        Args:
            provider: The SearchProvider implementation to use
            llm: The LLM instance for selection and synthesis (must implement LLMProtocol)
            search_type: Type identifier for results (e.g., "filesystem", "vector", "web")
        """
        self.provider: SearchProvider = provider
        self.llm: LLMProtocol = llm
        self.search_type: str = search_type
        self.abort_requested: bool = False
        self.search_id: Optional[str] = None

    async def _check_abort(self, request: Optional[Any] = None) -> bool:
        """
        Check if search should be aborted.

        Args:
            request: Optional FastAPI Request object for disconnect detection

        Returns:
            True if search should be aborted, False otherwise
        """
        if self.abort_requested:
            return True
        if request:
            return await request.is_disconnected()
        return False

    def _cancelled_result(
        self, query: str, phase: str, tool_logs: List[Dict[str, Any]]
    ) -> SearchResult:
        """Create a cancelled search result."""
        return SearchResult(
            success=False,
            message="Search cancelled.",
            data=SearchResultData(
                answer="Search was cancelled before completion.",
                sources=[],
                query=query,
                search_type=self.search_type,
                total_results=0,
                stats={"cancelled": True, "cancelled_at_phase": phase},
                tool_logs=tool_logs,
            ),
        )

    async def _llm_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        constrain_json: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: The prompt to send to the LLM
            system_message: Optional system message
            constrain_json: Optional JSON schema to constrain output format

        Returns:
            LLM response text
        """
        if not self.llm:
            raise ValueError("No LLM is loaded. Load a model first.")

        response = await self.llm.text_completion(
            prompt=prompt,
            system_message=system_message or "You are a helpful assistant.",
            stream=False,
            request=None,
            constrain_json_output=constrain_json,
        )

        # Collect response
        content = []
        async for item in response:
            content.append(item)

        # Parse response
        data = read_event_data(content)
        return data.get("text", "")

    async def _llm_select(
        self,
        items: List[SearchItem],
        query: str,
        max_select: int,
        phase_name: str = "preview",
    ) -> List[SearchItem]:
        """
        Use LLM to select the most relevant items.

        Args:
            items: List of items to select from
            query: The search query
            max_select: Maximum number of items to select
            phase_name: Name of the phase for logging

        Returns:
            List of selected items
        """
        if not items:
            return []

        if len(items) <= max_select:
            return items

        # Format items with indices for LLM (no sensitive info exposed)
        item_list_str = "\n".join(
            f"[{idx}] {item.name} ({item.type})"
            + (
                f" - {item.preview[:DEFAULT_CONTENT_PREVIEW_LENGTH]}..."
                if item.preview
                else ""
            )
            for idx, item in enumerate(items)
        )

        selection_prompt = f"""Given the user's query and a list of available items, select the most relevant ones.

User Query: {query}

Available Items:
{item_list_str}

Instructions:
1. Select up to {max_select} items that are most likely to contain information relevant to the query
2. Prioritize items by relevance to the query
3. Return ONLY a JSON array of item indices (the numbers in brackets)
4. Example: [0, 2, 5]"""

        try:
            selection_response = await self._llm_completion(
                prompt=selection_prompt,
                system_message="You are an item selection assistant. Return only a JSON array of integers.",
                constrain_json=SELECTION_SCHEMA,
            )

            # Parse the selection
            try:
                selected_indices = json.loads(selection_response)
                if isinstance(selected_indices, dict):
                    selected_indices = selected_indices.get("items", [])
            except json.JSONDecodeError:
                # Fallback to regex parsing
                json_match = re.search(r"\[.*\]", selection_response, re.DOTALL)
                if json_match:
                    selected_indices = json.loads(json_match.group())
                else:
                    selected_indices = list(range(min(max_select, len(items))))

            # Validate indices are within range
            selected_indices = [
                idx
                for idx in selected_indices
                if isinstance(idx, int) and 0 <= idx < len(items)
            ]

            return [items[idx] for idx in selected_indices[:max_select]]

        except Exception as e:
            # Log the error and fallback to first max_select items
            print(
                f"{common.PRNT_API} [AgenticSearch] LLM selection failed in {phase_name} phase: {e}",
                flush=True,
            )
            return items[:max_select]

    async def _synthesize(
        self, context: List[Dict[str, str]], query: str
    ) -> SearchResultData:
        """
        Generate final answer from collected context.

        Args:
            context: List of dicts with 'source' and 'content' keys
            query: The original search query

        Returns:
            SearchResultData with answer and sources
        """
        if not context:
            return SearchResultData(
                answer="Could not find relevant information to answer the query.",
                sources=[],
                query=query,
                search_type=self.search_type,
                total_results=0,
            )

        context_str = "\n\n---\n\n".join(
            f"Source: {c['source']}\n\n{c['content']}" for c in context
        )

        synthesis_prompt = f"""Based on the following content, answer the user's query.

User Query: {query}

Content:
{context_str}

Instructions:
1. Answer the query based ONLY on the information provided above
2. If the content doesn't contain relevant information, say so
3. Cite specific sources when making claims
4. Be concise but thorough"""

        try:
            answer = await self._llm_completion(
                prompt=synthesis_prompt,
                system_message="You are a helpful assistant that answers questions based on provided content. Always cite your sources.",
            )
        except Exception as e:
            answer = f"Failed to synthesize answer: {e}"

        # Build sources list
        sources = [
            SearchSource(
                id=c.get("source", "unknown"),
                type=self.search_type,
                name=c.get("source", "unknown"),
                snippet=(
                    c.get("content", "")[:DEFAULT_CONTENT_SNIPPET_LENGTH]
                    if c.get("content")
                    else None
                ),
            )
            for c in context
        ]

        return SearchResultData(
            answer=answer,
            sources=sources,
            query=query,
            search_type=self.search_type,
            total_results=len(sources),
        )

    async def search(
        self,
        query: str,
        initial_scope: Optional[str] = None,
        max_preview: int = DEFAULT_MAX_PREVIEW,
        max_read: int = DEFAULT_MAX_READ,
        max_expand: int = DEFAULT_MAX_EXPAND,
        auto_expand: bool = True,
        request: Optional[Any] = None,
        **kwargs: Any,
    ) -> SearchResult:
        """
        Execute the multi-phase agentic search.

        Args:
            query: The user's search query
            initial_scope: The initial scope to search (directory, collection, etc.)
                          If None, provider operates in discovery mode.
            max_preview: Maximum number of items to preview
            max_read: Maximum number of items to extract full content from
            max_expand: Maximum number of additional scopes to search during expansion
            auto_expand: Whether to automatically search additional scopes if needed
            request: Optional FastAPI Request for client disconnect detection
            **kwargs: Additional arguments passed to provider methods

        Returns:
            SearchResult with answer, sources, and stats
        """

        # Reset abort flag at start of each search
        self.abort_requested = False

        tool_logs = []
        all_context = []
        scopes_searched = [initial_scope] if initial_scope else []

        try:
            # Phase 1: DISCOVER
            scope_desc = (
                f"'{initial_scope}'" if initial_scope else "all (discovery mode)"
            )
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 1: Discovering items in scope {scope_desc}",
                flush=True,
            )
            items = await self.provider.discover(initial_scope, query=query, **kwargs)
            tool_logs.append(
                {
                    "phase": "discover",
                    "scope": initial_scope,
                    "items_found": len(items),
                }
            )

            if not items:
                return SearchResult(
                    success=True,
                    message="No items found in the specified scope.",
                    data=SearchResultData(
                        answer="No items were found in the specified scope.",
                        sources=[],
                        query=query,
                        search_type=self.search_type,
                        total_results=0,
                        stats={"scopes_searched": scopes_searched, "items_found": 0},
                        tool_logs=tool_logs,
                    ),
                )

            # Check for abort after Phase 1
            if await self._check_abort(request):
                return self._cancelled_result(query, "discover", tool_logs)

            # Phase 2: LLM SELECT for preview
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 2: Selecting items for preview",
                flush=True,
            )
            selected_for_preview = await self._llm_select(
                items, query, max_preview, "preview"
            )
            tool_logs.append(
                {
                    "phase": "select_preview",
                    "selected_count": len(selected_for_preview),
                }
            )

            # Check for abort after Phase 2
            if await self._check_abort(request):
                return self._cancelled_result(query, "select_preview", tool_logs)

            # Phase 3: PREVIEW
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 3: Getting previews",
                flush=True,
            )
            previewed = await self.provider.preview(selected_for_preview)
            tool_logs.append(
                {
                    "phase": "preview",
                    "previewed_count": len(previewed),
                }
            )

            # Check for abort after Phase 3
            if await self._check_abort(request):
                return self._cancelled_result(query, "preview", tool_logs)

            # Phase 4: LLM SELECT for extraction
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 4: Selecting items for extraction",
                flush=True,
            )
            selected_for_extract = await self._llm_select(
                previewed, query, max_read, "extract"
            )
            tool_logs.append(
                {
                    "phase": "select_extract",
                    "selected_count": len(selected_for_extract),
                }
            )

            # Check for abort after Phase 4
            if await self._check_abort(request):
                return self._cancelled_result(query, "select_extract", tool_logs)

            # Phase 5: EXTRACT
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 5: Extracting content",
                flush=True,
            )
            context = await self.provider.extract(selected_for_extract)
            all_context.extend(context)
            tool_logs.append(
                {
                    "phase": "extract",
                    "extracted_count": len(context),
                }
            )

            # Check for abort after Phase 5
            if await self._check_abort(request):
                return self._cancelled_result(query, "extract", tool_logs)

            # Phase 6: EXPAND (optional)
            if auto_expand and len(all_context) < max_read:
                print(
                    f"{common.PRNT_API} [AgenticSearch] Phase 6: Expanding search scope",
                    flush=True,
                )
                additional_scopes = self.provider.get_expandable_scopes(initial_scope)

                for scope in additional_scopes[:max_expand]:
                    if len(all_context) >= max_read:
                        break

                    # Check for abort during expansion
                    if await self._check_abort(request):
                        return self._cancelled_result(query, "expand", tool_logs)

                    scopes_searched.append(scope)
                    more_items = await self.provider.discover(
                        scope, query=query, **kwargs
                    )

                    if more_items:
                        more_selected = await self._llm_select(
                            more_items, query, max_preview, "expand_preview"
                        )
                        more_previewed = await self.provider.preview(more_selected)
                        more_to_extract = await self._llm_select(
                            more_previewed,
                            query,
                            max_read - len(all_context),
                            "expand_extract",
                        )
                        more_context = await self.provider.extract(more_to_extract)
                        all_context.extend(more_context)

                tool_logs.append(
                    {
                        "phase": "expand",
                        "additional_scopes": len(scopes_searched) - 1,
                        "total_context": len(all_context),
                    }
                )

            # Check for abort before synthesis
            if await self._check_abort(request):
                return self._cancelled_result(query, "pre_synthesize", tool_logs)

            # Phase 7: SYNTHESIZE
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 7: Synthesizing answer",
                flush=True,
            )
            result_data = await self._synthesize(all_context, query)
            result_data.stats = {
                "scopes_searched": scopes_searched,
                "items_discovered": len(items),
                "items_previewed": len(previewed),
                "items_extracted": len(all_context),
            }
            result_data.tool_logs = tool_logs

            return SearchResult(
                success=True,
                message="Search completed successfully.",
                data=result_data,
            )

        except Exception as e:
            print(
                f"{common.PRNT_API} [AgenticSearch] Search failed with error: {e}",
                flush=True,
            )
            return SearchResult(
                success=False,
                message=f"Search failed: {str(e)}",
                data=None,
            )
