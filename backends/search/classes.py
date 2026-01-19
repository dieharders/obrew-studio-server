from typing import List, Optional
from pydantic import BaseModel

# Import unified response types from base
from .base import SearchResult, SearchResultData, SearchSource


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
    collection: str  # Initial collection to search
    allowed_collections: List[str]  # Whitelist of collections for expansion
    top_k: Optional[int] = 50  # Max chunks to retrieve per collection
    max_preview: Optional[int] = 10  # Max chunks to preview
    max_extract: Optional[int] = 3  # Max chunks to extract full context from
    auto_expand: Optional[bool] = True  # Whether to search additional collections

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What are the key findings from the research?",
                    "collection": "research_papers",
                    "allowed_collections": ["research_papers", "literature_review"],
                    "top_k": 50,
                    "max_preview": 10,
                    "max_extract": 3,
                }
            ]
        }
    }


# Web Search classes
class WebSearchRequest(BaseModel):
    """Request for web search using DuckDuckGo."""

    query: str  # The search query
    website: Optional[str] = None  # Optional specific website to search
    allowed_domains: Optional[List[str]] = None  # Whitelist of domains (None = allow all)
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


# Keep legacy response type for backwards compatibility
class FileSystemSearchResponse(BaseModel):
    """Response from agentic file system search."""

    success: bool
    message: str
    data: Optional[dict] = None  # Contains: answer, sources, tool_logs, etc.


# Export unified response type for all endpoints
# All search endpoints should return SearchResult
__all__ = [
    "FileSystemSearchRequest",
    "FileSystemSearchResponse",
    "VectorSearchRequest",
    "WebSearchRequest",
    "SearchResult",
    "SearchResultData",
    "SearchSource",
]
