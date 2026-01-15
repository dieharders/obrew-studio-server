"""File scan tool for directory exploration with file metadata."""
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Scan a directory and return metadata about all files. Use this to get an overview of available files before deeper exploration."""

    directory_path: str = Field(
        ...,
        description="The directory path to scan.",
    )
    file_patterns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of file extensions to filter (e.g., ['.pdf', '.docx']). If not provided, returns all files.",
    )
    recursive: bool = Field(
        default=False,
        description="Whether to scan subdirectories recursively.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "directory_path": "/documents/reports",
                    "file_patterns": [".pdf", ".docx"],
                    "recursive": True,
                }
            ]
        }
    }


def _get_file_type(extension: str) -> str:
    """Map file extension to a human-readable type."""
    type_map = {
        ".pdf": "PDF Document",
        ".docx": "Word Document",
        ".doc": "Word Document (Legacy)",
        ".xlsx": "Excel Spreadsheet",
        ".xls": "Excel Spreadsheet (Legacy)",
        ".pptx": "PowerPoint Presentation",
        ".ppt": "PowerPoint Presentation (Legacy)",
        ".txt": "Text File",
        ".md": "Markdown",
        ".html": "HTML",
        ".htm": "HTML",
        ".json": "JSON",
        ".xml": "XML",
        ".csv": "CSV",
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".h": "C/C++ Header",
        ".rs": "Rust",
        ".go": "Go",
        ".rb": "Ruby",
        ".php": "PHP",
        ".sql": "SQL",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".ini": "Config (INI)",
        ".cfg": "Config",
        ".log": "Log File",
        ".rtf": "Rich Text Format",
    }
    return type_map.get(extension.lower(), f"File ({extension})")


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


async def main(**kwargs) -> List[Dict[str, Any]]:
    """
    Scan a directory and return metadata for all files.
    Returns list of dicts with: filename, path, type, size, size_bytes, modified, extension
    """
    directory_path = kwargs.get("directory_path", ".")
    file_patterns = kwargs.get("file_patterns")
    recursive = kwargs.get("recursive", False)

    # Ensure directory exists
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Convert patterns to lowercase set for filtering
    filter_extensions = None
    if file_patterns:
        filter_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in file_patterns}

    results = []

    # Walk or list directory
    if recursive:
        for root, _, files in os.walk(dir_path):
            for filename in files:
                file_path = Path(root) / filename
                ext = file_path.suffix.lower()

                # Skip if filter is set and extension doesn't match
                if filter_extensions and ext not in filter_extensions:
                    continue

                try:
                    stat = file_path.stat()
                    results.append({
                        "filename": filename,
                        "path": str(file_path.resolve()),
                        "relative_path": str(file_path.relative_to(dir_path)),
                        "type": _get_file_type(ext),
                        "extension": ext,
                        "size": _format_size(stat.st_size),
                        "size_bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    })
                except (OSError, PermissionError):
                    # Skip files we can't access
                    continue
    else:
        for item in dir_path.iterdir():
            if not item.is_file():
                continue

            ext = item.suffix.lower()
            if filter_extensions and ext not in filter_extensions:
                continue

            try:
                stat = item.stat()
                results.append({
                    "filename": item.name,
                    "path": str(item.resolve()),
                    "relative_path": item.name,
                    "type": _get_file_type(ext),
                    "extension": ext,
                    "size": _format_size(stat.st_size),
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except (OSError, PermissionError):
                continue

    # Sort by filename
    results.sort(key=lambda x: x["filename"].lower())

    return results
