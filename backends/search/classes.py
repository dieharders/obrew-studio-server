from typing import List, Optional, Dict, Any
from pydantic import BaseModel

# Import response types from the harness
from .harness import SearchResult, SearchResultData, SearchSource


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


# Structured Search classes
class StructuredItem(BaseModel):
    """Individual item in a structured search request."""

    id: Optional[str] = None  # Auto-generated if not provided
    name: Optional[str] = None  # Defaults to "Item {index}"
    content: str  # Required: The searchable text content
    metadata: Optional[Dict[str, Any]] = None


class StructuredSearchRequest(BaseModel):
    """Request for structured data search.

    Enables searching over ephemeral data sent by frontend applications.
    The data exists only for the duration of the request.
    """

    query: str  # The search query
    items: List[StructuredItem]  # The data to search over
    item_type: Optional[str] = "item"  # Type label (e.g., "conversation", "memo")
    max_preview: Optional[int] = 10  # Max items to preview
    max_extract: Optional[int] = 3  # Max items to extract from

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
                        },
                        {
                            "id": "msg-002",
                            "name": "Bob",
                            "content": "Agreed. We can use the jose library for JWT handling.",
                        },
                    ],
                    "item_type": "conversation",
                    "max_preview": 10,
                    "max_extract": 3,
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
