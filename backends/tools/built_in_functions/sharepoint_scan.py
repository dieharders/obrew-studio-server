"""SharePoint file list tool for listing pre-fetched SharePoint file metadata."""
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """List SharePoint files from pre-fetched metadata. The file list must be provided as a JSON array by the frontend."""

    files_json: str = Field(
        ...,
        description="JSON array of SharePoint file metadata objects. Each object should have: name, size, web_url, mime_type, last_modified, last_modified_by.",
    )
    filter_name: Optional[str] = Field(
        default=None,
        description="Optional substring filter for file names (case-insensitive).",
    )
    filter_type: Optional[str] = Field(
        default=None,
        description="Optional filter for MIME type (case-insensitive substring match).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "files_json": '[{"name": "report.docx", "size": 45000, "web_url": "https://contoso.sharepoint.com/report.docx", "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "last_modified": "2025-12-15T10:30:00Z", "last_modified_by": "John Smith"}]',
                },
                {
                    "files_json": '[{"name": "data.csv", "size": 12000}]',
                    "filter_name": "data",
                },
            ]
        }
    }


def _format_size(bytes_val: int) -> str:
    """Format byte count to human-readable string."""
    if bytes_val == 0:
        return "0 B"
    sizes = ["B", "KB", "MB", "GB"]
    i = 0
    val = float(bytes_val)
    while val >= 1024 and i < len(sizes) - 1:
        val /= 1024
        i += 1
    return f"{val:.1f} {sizes[i]}"


async def main(**kwargs) -> Dict[str, Any]:
    """
    List SharePoint files from pre-fetched metadata JSON.
    Returns dict with: files (list), total_count, filtered_count
    """
    files_json = kwargs.get("files_json")
    filter_name = kwargs.get("filter_name")
    filter_type = kwargs.get("filter_type")

    if not files_json:
        raise ValueError("files_json is required")

    try:
        files = json.loads(files_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in files_json: {e}")

    if not isinstance(files, list):
        raise ValueError("files_json must be a JSON array")

    total_count = len(files)

    # Apply filters
    filtered = files
    if filter_name:
        filter_name_lower = filter_name.lower()
        filtered = [
            f for f in filtered
            if filter_name_lower in str(f.get("name", "")).lower()
        ]
    if filter_type:
        filter_type_lower = filter_type.lower()
        filtered = [
            f for f in filtered
            if filter_type_lower in str(f.get("mime_type", "")).lower()
        ]

    # Format output
    formatted_files = []
    for f in filtered:
        entry = {
            "name": f.get("name", "(unnamed)"),
            "size": _format_size(f.get("size", 0)),
            "web_url": f.get("web_url", ""),
            "mime_type": f.get("mime_type", ""),
            "last_modified": f.get("last_modified", ""),
            "last_modified_by": f.get("last_modified_by", ""),
        }
        formatted_files.append(entry)

    return {
        "files": formatted_files,
        "total_count": total_count,
        "filtered_count": len(filtered),
    }
