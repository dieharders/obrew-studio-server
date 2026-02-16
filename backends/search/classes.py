import json
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from .harness import (
    SearchResult,
    SearchResultData,
    SearchSource,
    DEFAULT_MAX_PREVIEW,
    DEFAULT_MAX_READ,
    DEFAULT_MAX_EXPAND,
    DEFAULT_MAX_DISCOVER_ITEMS,
)

# Limits for email search
MAX_EMAIL_ITEMS = 500  # Maximum number of emails per request

# Limits for structured search to prevent abuse
MAX_STRUCTURED_ITEMS = 1000  # Maximum number of items
MAX_CONTENT_DEPTH = 10  # Maximum nesting depth for content
MAX_TOTAL_PAYLOAD_SIZE_MB = 50  # Maximum total payload size in MB
MAX_DEPTH_CHECK_ITERATIONS = 100  # Max items to check during depth validation


# Agentic File System Search classes
class FileSystemSearchRequest(BaseModel):
    """Request for agentic file system search."""

    query: str  # The search query
    directories: List[str]  # Directories to search in (also serves as whitelist)
    file_patterns: Optional[List[str]] = (
        None  # File extensions to filter (e.g., [".pdf", ".docx"])
    )
    max_preview: Optional[int] = DEFAULT_MAX_PREVIEW  # Max files to preview
    max_read: Optional[int] = DEFAULT_MAX_READ  # Max files to fully read/parse
    max_iterations: Optional[int] = DEFAULT_MAX_EXPAND  # Max directories to search (for expansion)
    auto_expand: Optional[bool] = True  # Whether to search additional directories

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find all documents about quarterly sales reports",
                    "directories": [
                        "/documents/reports",
                        "/documents/archives",
                    ],
                    "file_patterns": [".pdf", ".docx"],
                    "max_preview": 10,
                    "max_read": 3,
                }
            ]
        }
    }


# Vector/Embedding Search classes
class VectorSearchRequest(BaseModel):
    """Request for vector/embedding collection search."""

    query: str  # The search query
    collections: Optional[List[str]] = None  # Optional - if None/empty, discover all
    top_k: Optional[int] = DEFAULT_MAX_DISCOVER_ITEMS  # Max chunks to retrieve per collection
    max_preview: Optional[int] = DEFAULT_MAX_PREVIEW  # Max collections/chunks to preview
    max_read: Optional[int] = DEFAULT_MAX_READ  # Max collections/chunks to read/extract from
    auto_expand: Optional[bool] = True  # Whether to search additional collections

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the key findings from the research?",
                    "collections": ["research_papers", "technical_docs"],
                    "top_k": 50,
                    "max_preview": 10,
                    "max_read": 3,
                },
                {
                    "query": "Find information about machine learning",
                    "top_k": 50,
                    "max_preview": 10,
                    "max_read": 3,
                },
            ]
        }
    }


# Web Search classes
class WebSearchRequest(BaseModel):
    """Request for web search using DuckDuckGo."""

    query: str  # The search query
    website: Optional[List[str]] = (
        None  # Domain filter: None/[] = all, [one] = single site, [many] = whitelist
    )
    max_preview: Optional[int] = DEFAULT_MAX_PREVIEW  # Max URLs to preview
    max_read: Optional[int] = DEFAULT_MAX_READ  # Max pages to read/extract full content from

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Python asyncio best practices",
                    "website": ["docs.python.org", "stackoverflow.com"],
                    "max_preview": 10,
                    "max_read": 3,
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
        return max(_check_content_depth(v, current_depth + 1) for v in obj.values())
    elif isinstance(obj, list):
        if not obj:
            return current_depth
        return max(
            _check_content_depth(item, current_depth + 1)
            for item in obj[:MAX_DEPTH_CHECK_ITERATIONS]
        )
    else:
        return current_depth


def _estimate_content_size(obj: Any) -> int:
    """Estimate the size of content in bytes."""
    if isinstance(obj, str):
        return len(obj.encode("utf-8", errors="ignore"))
    elif isinstance(obj, (dict, list)):
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
    content: Union[
        str, Dict[str, Any], List[Any], int, float, bool, None
    ]  # Explicit types
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
    max_preview: Optional[int] = DEFAULT_MAX_PREVIEW  # Max items to preview
    max_read: Optional[int] = DEFAULT_MAX_READ  # Max items to read/extract from
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
                            "content": {
                                "text": "Agreed. We can use jose library.",
                                "attachments": [],
                            },
                            "metadata": {"channel": "engineering"},
                        },
                    ],
                    "max_preview": 10,
                    "max_read": 3,
                    "group_by": "channel",
                    "auto_expand": True,
                }
            ]
        }
    }


# Email Search classes
class EmailSearchRequest(BaseModel):
    """Request for agentic email search.

    Searches over email data sent by the frontend (fetched from MS Graph API).
    The emails exist only for the duration of the request.

    The search uses the same multi-phase agentic pattern as filesystem search:
    discover (metadata) → preview (bodyPreview) → extract (full body) → synthesize.
    """

    query: str  # The search query
    emails: List[Dict[str, Any]]  # Raw email objects (Microsoft Graph API format)
    max_preview: Optional[int] = DEFAULT_MAX_PREVIEW  # Max emails to preview
    max_read: Optional[int] = DEFAULT_MAX_READ  # Max emails to read fully
    auto_expand: Optional[bool] = False  # Group by conversationId and expand to other threads

    @field_validator("emails")
    @classmethod
    def validate_emails_count(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate that email count doesn't exceed maximum."""
        if len(v) > MAX_EMAIL_ITEMS:
            raise ValueError(
                f"Number of emails ({len(v)}) exceeds maximum of {MAX_EMAIL_ITEMS}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Summarize information about shareholders",
                    "emails": [
                        {
                            "id": "AAMkAGI2...",
                            "subject": "Q4 Shareholder Report",
                            "from": {"emailAddress": {"name": "John", "address": "john@example.com"}},
                            "bodyPreview": "Please find attached the Q4 shareholder report...",
                            "receivedDateTime": "2025-12-15T10:30:00Z",
                        }
                    ],
                    "max_preview": 10,
                    "max_read": 3,
                }
            ]
        }
    }


# SharePoint Search classes
MAX_SHAREPOINT_ITEMS = 500  # Maximum number of SharePoint files per request
MAX_SHAREPOINT_CONTENT_LENGTH = 102_400  # Maximum content length per file (~100KB)


class SharePointSearchItem(BaseModel):
    """Individual item in a SharePoint search request."""

    id: str
    name: str
    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_SHAREPOINT_CONTENT_LENGTH,
        description="Text content fetched by frontend from Graph API. Must be non-empty.",
    )
    web_url: Optional[str] = None
    mime_type: Optional[str] = None
    drive_id: Optional[str] = None
    last_modified: Optional[str] = None
    last_modified_by: Optional[str] = None


class SharePointSearchRequest(BaseModel):
    """Request for agentic SharePoint file search.

    Searches over SharePoint file data sent by the frontend (fetched from MS Graph API).
    The data exists only for the duration of the request.

    The search uses the same multi-phase agentic pattern:
    discover (metadata) → preview (content snippet) → extract (full content) → synthesize.
    """

    query: str  # The search query
    items: List[SharePointSearchItem]  # SharePoint file data from frontend
    max_preview: Optional[int] = DEFAULT_MAX_PREVIEW  # Max files to preview
    max_read: Optional[int] = DEFAULT_MAX_READ  # Max files to read fully

    @field_validator("items")
    @classmethod
    def validate_items_count(
        cls, v: List[SharePointSearchItem],
    ) -> List[SharePointSearchItem]:
        """Validate that item count doesn't exceed maximum."""
        if len(v) > MAX_SHAREPOINT_ITEMS:
            raise ValueError(
                f"Number of items ({len(v)}) exceeds maximum of {MAX_SHAREPOINT_ITEMS}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find the latest project proposal document",
                    "items": [
                        {
                            "id": "01ABCDEF...",
                            "name": "Project Proposal v2.docx",
                            "content": "This document outlines the proposed project timeline...",
                            "web_url": "https://contoso.sharepoint.com/sites/team/Documents/proposal.docx",
                            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            "last_modified": "2025-12-15T10:30:00Z",
                            "last_modified_by": "John Smith",
                        }
                    ],
                    "max_preview": 10,
                    "max_read": 3,
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
    "EmailSearchRequest",
    "SharePointSearchItem",
    "SharePointSearchRequest",
    "SearchResult",
    "SearchResultData",
    "SearchSource",
]
