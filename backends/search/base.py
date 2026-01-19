"""
Base classes for the unified search architecture.

This module provides the core abstractions for a generalized multi-phase search loop
that works across different context types (file system, vector/embedding, web).

The search loop follows these phases:
1. DISCOVER - Find items in scope (directory, collection, search query)
2. LLM SELECT - Use LLM to select relevant items for preview
3. PREVIEW - Get preview content for selected items
4. RERANK - (Optional) Re-rank items by relevance
5. LLM DEEP SELECT - Use LLM to select items for full extraction
6. EXTRACT - Get full content for selected items
7. EXPAND - Search additional scopes if needed
8. SYNTHESIZE - Generate final answer from context
"""

import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


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

    async def rerank(self, items: List[SearchItem], query: str) -> List[SearchItem]:
        """
        Optional: Re-rank items by relevance to the query.

        Default implementation is a no-op (returns items as-is).
        Override in subclass to implement re-ranking (e.g., cross-encoder,
        LLM-based scoring, reciprocal rank fusion).

        Args:
            items: List of items to re-rank
            query: The search query

        Returns:
            Re-ranked list of items
        """
        return items


class AgenticSearch:
    """
    Orchestrates the multi-phase search loop for any SearchProvider.

    This class implements the generalized agentic search pattern that works
    across different context types. It uses LLM-based selection at multiple
    stages to progressively narrow down the search results.
    """

    def __init__(self, provider: SearchProvider, llm, search_type: str):
        """
        Initialize the AgenticSearch orchestrator.

        Args:
            provider: The SearchProvider implementation to use
            llm: The LLM instance for selection and synthesis
            search_type: Type identifier for results (e.g., "filesystem", "vector", "web")
        """
        self.provider = provider
        self.llm = llm
        self.search_type = search_type

    async def _llm_completion(
        self,
        prompt: str,
        system_message: str = None,
        constrain_json: dict = None,
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
        from inference.helpers import read_event_data

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
            + (f" - {item.preview[:100]}..." if item.preview else "")
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

        except Exception:
            # Fallback: return first max_select items
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
                snippet=c.get("content", "")[:200] if c.get("content") else None,
            )
            for c in context
        ]

        return SearchResultData(
            answer=answer,
            sources=sources,
            query=query,
            search_type=self.search_type,
        )

    async def search(
        self,
        query: str,
        initial_scope: str,
        max_preview: int = 10,
        max_extract: int = 3,
        auto_expand: bool = True,
        **kwargs,
    ) -> SearchResult:
        """
        Execute the multi-phase agentic search.

        Args:
            query: The user's search query
            initial_scope: The initial scope to search (directory, collection, etc.)
            max_preview: Maximum number of items to preview
            max_extract: Maximum number of items to extract full content from
            auto_expand: Whether to automatically search additional scopes if needed
            **kwargs: Additional arguments passed to provider methods

        Returns:
            SearchResult with answer, sources, and stats
        """
        from core import common

        tool_logs = []
        all_context = []
        scopes_searched = [initial_scope]

        try:
            # Phase 1: DISCOVER
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 1: Discovering items in scope",
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
                        stats={"scopes_searched": scopes_searched, "items_found": 0},
                        tool_logs=tool_logs,
                    ),
                )

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

            # Phase 4: RERANK (optional)
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 4: Re-ranking (if implemented)",
                flush=True,
            )
            reranked = await self.provider.rerank(previewed, query)

            # Phase 5: LLM SELECT for extraction
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 5: Selecting items for extraction",
                flush=True,
            )
            selected_for_extract = await self._llm_select(
                reranked, query, max_extract, "extract"
            )
            tool_logs.append(
                {
                    "phase": "select_extract",
                    "selected_count": len(selected_for_extract),
                }
            )

            # Phase 6: EXTRACT
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 6: Extracting content",
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

            # Phase 7: EXPAND (optional)
            if auto_expand and len(all_context) < max_extract:
                print(
                    f"{common.PRNT_API} [AgenticSearch] Phase 7: Expanding search scope",
                    flush=True,
                )
                additional_scopes = self.provider.get_expandable_scopes(initial_scope)

                for scope in additional_scopes[:2]:  # Limit expansion
                    if len(all_context) >= max_extract:
                        break

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
                            max_extract - len(all_context),
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

            # Phase 8: SYNTHESIZE
            print(
                f"{common.PRNT_API} [AgenticSearch] Phase 8: Synthesizing answer",
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
            return SearchResult(
                success=False,
                message=f"Search failed: {str(e)}",
                data=None,
            )
