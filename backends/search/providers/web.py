"""
WebProvider - Search provider for web search using DuckDuckGo.

This provider implements the SearchProvider protocol for searching
the web using DuckDuckGo and extracting content from web pages.
"""

import asyncio
from typing import List, Dict, Optional
from urllib.parse import urlparse
from ddgs import DDGS
from ..harness import (
    SearchProvider,
    SearchItem,
    DEFAULT_CONTENT_EXTRACT_LENGTH,
    DEFAULT_CONTENT_SNIPPET_LENGTH,
)

# Rate limiting constants
DEFAULT_REQUEST_DELAY = 0.5  # Seconds between requests
MAX_CONCURRENT_REQUESTS = 3  # Maximum concurrent page fetches


class WebProvider(SearchProvider):
    """
    Search provider for web search using DuckDuckGo.

    Uses duckduckgo-search library for search and httpx + beautifulsoup4
    for content extraction. Supports domain whitelisting for security.
    """

    def __init__(
        self,
        app,
        website: Optional[List[str]] = None,
        max_pages: int = 10,
    ):
        """
        Initialize the WebProvider.

        Args:
            app: FastAPI application instance
            website: Domain filter - None/[] = search all, [one] = single site, [many] = whitelist
            max_pages: Maximum number of pages to fetch content from
        """
        self.app = app
        self.website = website or []
        self.max_pages = max_pages
        self._ddgs = None
        self._http_client = None
        self._request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self._last_request_time = 0.0

    def _get_ddgs(self):
        """Lazy-load the DuckDuckGo search client."""
        if self._ddgs is None:
            self._ddgs = DDGS()
        return self._ddgs

    async def _get_http_client(self):
        """Lazy-load the async HTTP client."""
        if self._http_client is None:
            import httpx

            self._http_client = httpx.AsyncClient(
                timeout=10.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                },
            )
        return self._http_client

    def _is_allowed_url(self, url: str) -> bool:
        """
        Check if a URL is in the allowed domains list.

        Args:
            url: URL to check

        Returns:
            True if URL is allowed, False otherwise
        """
        if not self.website:
            return True  # No whitelist = allow all

        try:
            domain = urlparse(url).netloc.lower()
            return any(
                domain == allowed.lower() or domain.endswith(f".{allowed.lower()}")
                for allowed in self.website
            )
        except Exception:
            return False

    async def discover(self, scope: str, **kwargs) -> List[SearchItem]:
        """
        Discover URLs via DuckDuckGo search.

        Args:
            scope: The search query
            **kwargs: Additional arguments (query can also be passed here)

        Returns:
            List of SearchItem objects representing discovered URLs
        """
        from core import common

        # The scope for web search is the query itself
        query = kwargs.get("query", scope)
        max_results = kwargs.get("max_results", 20)

        # Build search query with site filter
        if len(self.website) == 1:
            # Single website - search only that site
            search_query = f"site:{self.website[0]} {query}"
        elif len(self.website) > 1:
            # Multiple domains - search across whitelist
            domain_filter = " OR ".join(f"site:{d}" for d in self.website)
            search_query = f"{query} ({domain_filter})"
        else:
            # No restrictions
            search_query = query

        try:
            ddgs = self._get_ddgs()
            results = list(ddgs.text(search_query, max_results=max_results))

            items = []
            for i, r in enumerate(results):
                url = r.get("href", "")

                # Validate URL is in allowed domains (double-check)
                if not self._is_allowed_url(url):
                    continue

                items.append(
                    SearchItem(
                        id=url,
                        name=r.get("title", f"Result {i}"),
                        type="url",
                        preview=r.get("body", "")[:DEFAULT_CONTENT_SNIPPET_LENGTH],
                        metadata={
                            "href": url,
                            "title": r.get("title", ""),
                            "body": r.get("body", ""),
                        },
                        requires_extraction=True,  # Web pages need content fetching
                    )
                )

            print(
                f"{common.PRNT_API} [WebProvider] Found {len(items)} URLs for query: {query[:50]}...",
                flush=True,
            )

            return items

        except Exception as e:
            raise ValueError(f"Failed to search web: {e}")

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given URLs.

        For web search, items already have preview (snippet) from search results.

        Args:
            items: List of URL items to preview

        Returns:
            Same items (preview already populated from search snippets)
        """
        # URLs already have preview (body/snippet) from DuckDuckGo results
        return items

    async def _rate_limit(self):
        """
        Apply rate limiting by waiting if necessary.

        Ensures minimum delay between requests to avoid overwhelming servers.
        """
        import time

        current_time = time.time()
        elapsed = current_time - self._last_request_time

        if elapsed < DEFAULT_REQUEST_DELAY:
            await asyncio.sleep(DEFAULT_REQUEST_DELAY - elapsed)

        self._last_request_time = time.time()

    async def _fetch_page_content(self, url: str) -> str:
        """
        Fetch and extract text content from a webpage with rate limiting.

        Args:
            url: URL to fetch

        Returns:
            Extracted text content
        """
        from bs4 import BeautifulSoup

        # Apply rate limiting with semaphore for concurrency control
        async with self._request_semaphore:
            await self._rate_limit()

            client = await self._get_http_client()
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, and navigation elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return "\n".join(lines)

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given URLs.

        Fetches page content and extracts text.

        Args:
            items: List of URL items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        from core import common

        context = []

        for item in items[: self.max_pages]:  # Limit to max_pages
            try:
                content = await self._fetch_page_content(item.id)

                context.append(
                    {
                        "source": f"[{item.name}]({item.id})",
                        "content": content[:DEFAULT_CONTENT_EXTRACT_LENGTH],
                    }
                )

                print(
                    f"{common.PRNT_API} [WebProvider] Extracted content from: {item.id[:50]}...",
                    flush=True,
                )

            except Exception as e:
                print(
                    f"{common.PRNT_API} [WebProvider] Failed to fetch {item.id}: {e}",
                    flush=True,
                )
                # Still include the preview if we couldn't fetch full content
                if item.preview:
                    context.append(
                        {
                            "source": f"[{item.name}]({item.id}) [preview only]",
                            "content": item.preview,
                        }
                    )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return additional search scopes.

        For web search, expansion means running additional searches,
        which is handled differently (not through scope expansion).

        Args:
            current_scope: The query that was just searched

        Returns:
            Empty list (pagination handled separately)
        """
        # Web search doesn't use scope expansion in the same way
        # Pagination would be handled by increasing max_results
        return []

    async def close(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
