from typing import List, Optional
from pydantic import BaseModel


# Agentic File Search classes
class FileSearchRequest(BaseModel):
    """Request for agentic file search."""

    query: str  # The search query
    directory: str  # Directory to search in
    allowed_directories: List[str]  # Whitelist of directories the agent can access
    file_patterns: Optional[List[str]] = None  # File extensions to filter (e.g., [".pdf", ".docx"])
    max_files_preview: Optional[int] = 10  # Max files to preview
    max_files_parse: Optional[int] = 3  # Max files to fully parse

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find all documents about quarterly sales reports",
                    "directory": "/documents/reports",
                    "allowed_directories": ["/documents/reports", "/documents/archives"],
                    "file_patterns": [".pdf", ".docx"],
                    "max_files_preview": 10,
                    "max_files_parse": 3,
                }
            ]
        }
    }


class FileSearchResponse(BaseModel):
    """Response from agentic file search."""

    success: bool
    message: str
    data: Optional[dict] = None  # Contains: answer, sources, tool_logs, etc.
