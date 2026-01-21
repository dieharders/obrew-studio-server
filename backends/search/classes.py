from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, field_validator, model_validator

# Import response types from the harness
from .harness import SearchResult, SearchResultData, SearchSource

# Limits for structured search to prevent abuse
MAX_STRUCTURED_ITEMS = 1000  # Maximum number of items
MAX_CONTENT_DEPTH = 10  # Maximum nesting depth for content
MAX_TOTAL_PAYLOAD_SIZE_MB = 50  # Maximum total payload size in MB


# Agentic File System Search classes
class FileSystemSearchRequest(BaseModel):
    """Request for agentic file system search."""

    query: str  # The search query
    directory: str  # Directory to search in
    allowed_directories: List[str]  # Whitelist of directories the agent can access
    file_patterns: Optional[List[str]] = (
        None  # File extensions to filter (e.g., [".pdf", ".docx"])
    )
    max_files_preview: Optional[int] = 10  # Max files to preview
    max_files_parse: Optional[int] = 3  # Max files to fully parse
    max_iterations: Optional[int] = 3  # Max directories to search (for expansion)
    auto_expand: Optional[bool] = True  # Whether to search additional directories

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find all documents about quarterly sales reports",
                    "directory": "/documents/reports",
                    "allowed_directories": [
                        "/documents/reports",
                        "/documents/archives",
                    ],
                    "file_patterns": [".pdf", ".docx"],
                    "max_files_preview": 10,
                    "max_files_parse": 3,
                }
            ]
        }
    }


# Vector/Embedding Search classes
class VectorSearchRequest(BaseModel):
    """Request for vector/embedding collection search."""

    query: str  # The search query
    collections: Optional[List[str]] = None  # Optional - if None/empty, discover all
    top_k: Optional[int] = 50  # Max chunks to retrieve per collection
    max_preview: Optional[int] = 10  # Max collections/chunks to preview
    max_extract: Optional[int] = 3  # Max collections/chunks to extract from
    auto_expand: Optional[bool] = True  # Whether to search additional collections

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the key findings from the research?",
                    "collections": ["research_papers", "technical_docs"],
                    "top_k": 50,
                    "max_preview": 10,
                    "max_extract": 3,
                },
                {
                    "query": "Find information about machine learning",
                    "top_k": 50,
                    "max_preview": 10,
                    "max_extract": 3,
                },
            ]
        }
    }


# Web Search classes
class WebSearchRequest(BaseModel):
    """Request for web search using DuckDuckGo."""

    query: str  # The search query
    website: Optional[str] = None  # Optional specific website to search
    allowed_domains: Optional[List[str]] = (
        None  # Whitelist of domains (None = allow all)
    )
    max_pages: Optional[int] = 10  # Max pages to fetch content from
    max_preview: Optional[int] = 10  # Max URLs to preview
    max_extract: Optional[int] = 3  # Max pages to extract full content from

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Python asyncio best practices",
                    "website": "docs.python.org",
                    "allowed_domains": ["docs.python.org", "stackoverflow.com"],
                    "max_pages": 10,
                    "max_preview": 10,
                    "max_extract": 3,
                }
            ]
        }
    }


def _check_content_depth(obj: Any, current_depth: int = 0) -> int:
    """
    Recursively check the depth of nested content.

    Returns the maximum depth found, raises ValueError if max depth exceeded.
    """
    if current_depth > MAX_CONTENT_DEPTH:
        raise ValueError(
            f"Content nesting depth exceeds maximum of {MAX_CONTENT_DEPTH}"
        )

    if isinstance(obj, dict):
        if not obj:
            return current_depth
        return max(
            _check_content_depth(v, current_depth + 1) for v in obj.values()
        )
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(
            _check_content_depth(item, current_depth + 1) for item in obj[:100]  # Limit iteration
        )
    else:
        return current_depth


def _estimate_content_size(obj: Any) -> int:
    """Estimate the size of content in bytes."""
    if isinstance(obj, str):
        return len(obj.encode("utf-8", errors="ignore"))
    elif isinstance(obj, (dict, list)):
        import json

        try:
            return len(json.dumps(obj, default=str).encode("utf-8", errors="ignore"))
        except (TypeError, ValueError):
            return len(str(obj).encode("utf-8", errors="ignore"))
    else:
        return len(str(obj).encode("utf-8", errors="ignore"))


# Structured Search classes
class StructuredItem(BaseModel):
    """Individual item in a structured search request."""

    id: Optional[str] = None  # Auto-generated if not provided
    name: Optional[str] = None  # Defaults to "Item {index}"
    content: Union[str, Dict[str, Any], List[Any], int, float, bool, None]  # Explicit types
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("content")
    @classmethod
    def validate_content_depth(cls, v: Any) -> Any:
        """Validate that content doesn't exceed maximum nesting depth."""
        try:
            _check_content_depth(v)
        except ValueError as e:
            raise ValueError(str(e))
        return v


class StructuredSearchRequest(BaseModel):
    """Request for structured data search.

    Enables searching over ephemeral data sent by frontend applications.
    The data exists only for the duration of the request.

    Limits:
    - Maximum {MAX_STRUCTURED_ITEMS} items
    - Maximum {MAX_TOTAL_PAYLOAD_SIZE_MB}MB total payload size
    - Maximum content nesting depth of {MAX_CONTENT_DEPTH}
    """

    query: str  # The search query
    items: List[StructuredItem]  # The data to search over
    max_preview: Optional[int] = 10  # Max items to preview
    max_extract: Optional[int] = 3  # Max items to extract from
    group_by: Optional[str] = None  # Metadata field to group items by for expansion
    auto_expand: Optional[bool] = False  # Whether to search additional groups

    @field_validator("items")
    @classmethod
    def validate_items_count(cls, v: List[StructuredItem]) -> List[StructuredItem]:
        """Validate that items count doesn't exceed maximum."""
        if len(v) > MAX_STRUCTURED_ITEMS:
            raise ValueError(
                f"Number of items ({len(v)}) exceeds maximum of {MAX_STRUCTURED_ITEMS}"
            )
        return v

    @model_validator(mode="after")
    def validate_total_payload_size(self) -> "StructuredSearchRequest":
        """Validate that total payload size doesn't exceed maximum."""
        total_size = 0
        for item in self.items:
            total_size += _estimate_content_size(item.content)
            if item.metadata:
                total_size += _estimate_content_size(item.metadata)

        max_bytes = MAX_TOTAL_PAYLOAD_SIZE_MB * 1024 * 1024
        if total_size > max_bytes:
            size_mb = total_size / (1024 * 1024)
            raise ValueError(
                f"Total payload size ({size_mb:.1f}MB) exceeds maximum of {MAX_TOTAL_PAYLOAD_SIZE_MB}MB"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What did we decide about authentication?",
                    "items": [
                        {
                            "id": "msg-001",
                            "name": "Alice",
                            "content": "I think we should use JWT tokens for authentication.",
                            "metadata": {"channel": "engineering"},
                        },
                        {
                            "id": "msg-002",
                            "name": "Bob",
                            "content": {"text": "Agreed. We can use jose library.", "attachments": []},
                            "metadata": {"channel": "engineering"},
                        },
                    ],
                    "max_preview": 10,
                    "max_extract": 3,
                    "group_by": "channel",
                    "auto_expand": True,
                }
            ]
        }
    }


# Export unified response type for all endpoints
# All search endpoints should return SearchResult
__all__ = [
    "FileSystemSearchRequest",
    "VectorSearchRequest",
    "WebSearchRequest",
    "StructuredItem",
    "StructuredSearchRequest",
    "SearchResult",
    "SearchResultData",
    "SearchSource",
]
