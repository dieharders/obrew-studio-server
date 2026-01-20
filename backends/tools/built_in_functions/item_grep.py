"""Item grep tool for pattern-based search across structured items."""
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Search for a pattern across all structured items, searching content (strings, keys, values) and metadata."""

    pattern: str = Field(
        ...,
        description="The search pattern (plain text or regular expression).",
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of structured items to search. Each item should have id, name, content, and optional metadata.",
    )
    search_metadata: bool = Field(
        default=True,
        description="Whether to also search in item metadata.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the search should be case-sensitive.",
    )
    use_regex: bool = Field(
        default=False,
        description="Whether to interpret the pattern as a regular expression.",
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of matching items to return.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pattern": "auth",
                    "items": [
                        {
                            "id": "config-001",
                            "name": "app_config",
                            "content": {"auth": {"provider": "jwt"}},
                        }
                    ],
                    "search_metadata": True,
                }
            ]
        }
    }


def _find_string_matches(
    pattern: re.Pattern, text: str, path: str = ""
) -> List[Dict[str, Any]]:
    """Find matches in a string, returning line-based results."""
    matches = []
    lines = text.split("\n")

    for line_num, line in enumerate(lines, 1):
        if pattern.search(line):
            matches.append({
                "path": path or "(content)",
                "type": "string",
                "line_number": line_num,
                "match_line": line.strip()[:200],
            })

    return matches


def _find_object_matches(
    pattern: re.Pattern, obj: Any, current_path: str = ""
) -> List[Dict[str, Any]]:
    """Recursively find matches in an object structure (keys and values)."""
    matches = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{current_path}.{key}" if current_path else key

            # Check if key matches
            if pattern.search(str(key)):
                matches.append({
                    "path": key_path,
                    "type": "key",
                    "matched_key": key,
                })

            # Recursively check value
            matches.extend(_find_object_matches(pattern, value, key_path))

    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            item_path = f"{current_path}[{idx}]"
            matches.extend(_find_object_matches(pattern, item, item_path))

    elif isinstance(obj, str):
        if pattern.search(obj):
            matches.append({
                "path": current_path or "(value)",
                "type": "value",
                "matched_value": obj[:200] + ("..." if len(obj) > 200 else ""),
            })

    elif obj is not None:
        # Check numeric or other values converted to string
        str_val = str(obj)
        if pattern.search(str_val):
            matches.append({
                "path": current_path or "(value)",
                "type": "value",
                "matched_value": str_val,
            })

    return matches


async def main(**kwargs) -> Dict[str, Any]:
    """
    Search for a pattern across structured items.
    Returns dict with: pattern, matches, total_matches, items_searched
    """
    pattern_str = kwargs.get("pattern")
    items = kwargs.get("items", [])
    search_metadata = kwargs.get("search_metadata", True)
    case_sensitive = kwargs.get("case_sensitive", False)
    use_regex = kwargs.get("use_regex", False)
    max_results = kwargs.get("max_results", 50)

    if not pattern_str:
        raise ValueError("pattern is required")

    if not items:
        return {
            "pattern": pattern_str,
            "matches": [],
            "total_matches": 0,
            "items_searched": 0,
        }

    # Compile the pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if use_regex:
        try:
            pattern = re.compile(pattern_str, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    else:
        pattern = re.compile(re.escape(pattern_str), flags)

    all_matches = []
    items_with_matches = 0

    for idx, item in enumerate(items):
        if len(all_matches) >= max_results:
            break

        item_id = item.get("id") or f"item_{idx}"
        item_name = item.get("name") or f"Item {idx}"
        content = item.get("content")
        metadata = item.get("metadata", {})

        content_matches = []
        metadata_matches = []

        # Search content based on type
        if isinstance(content, str):
            content_matches = _find_string_matches(pattern, content)
        elif content is not None:
            content_matches = _find_object_matches(pattern, content)

        # Optionally search metadata
        if search_metadata and metadata:
            metadata_matches = _find_object_matches(pattern, metadata, "metadata")

        if content_matches or metadata_matches:
            items_with_matches += 1
            all_matches.append({
                "id": item_id,
                "name": item_name,
                "content_matches": content_matches[:10],  # Limit per item
                "metadata_matches": metadata_matches[:5],
                "total_content_matches": len(content_matches),
                "total_metadata_matches": len(metadata_matches),
            })

    return {
        "pattern": pattern_str,
        "matches": all_matches,
        "total_matches": items_with_matches,
        "items_searched": len(items),
        "truncated": len(all_matches) >= max_results,
    }
