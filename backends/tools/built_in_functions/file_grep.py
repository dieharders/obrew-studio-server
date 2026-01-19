"""File grep tool for pattern-based content searching."""
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Search for a pattern (text or regex) within files in a directory. Returns matching lines with context."""

    pattern: str = Field(
        ...,
        description="The search pattern (plain text or regular expression).",
    )
    directory_path: str = Field(
        ...,
        description="The directory to search in.",
    )
    file_patterns: Optional[List[str]] = Field(
        default=None,
        description="Optional list of file extensions to search (e.g., ['.txt', '.md']). If not provided, searches common text files.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the search should be case-sensitive.",
    )
    use_regex: bool = Field(
        default=False,
        description="Whether to interpret the pattern as a regular expression.",
    )
    context_lines: int = Field(
        default=1,
        description="Number of lines to show before and after each match.",
    )
    max_results: int = Field(
        default=50,
        description="Maximum number of matches to return.",
    )
    recursive: bool = Field(
        default=True,
        description="Whether to search subdirectories recursively.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pattern": "TODO",
                    "directory_path": "/projects/myapp",
                    "file_patterns": [".py", ".js"],
                    "recursive": True,
                },
                {
                    "pattern": r"def \w+\(",
                    "directory_path": "/projects/myapp/src",
                    "use_regex": True,
                },
            ]
        }
    }


# Default text file extensions to search
DEFAULT_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".xml", ".yaml", ".yml",
    ".csv", ".log", ".ini", ".cfg", ".conf", ".properties",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".html", ".htm", ".css", ".scss", ".sass", ".sql",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".r", ".R", ".jl", ".lua", ".pl", ".toml", ".env",
}


def _should_search_file(file_path: Path, filter_extensions: Optional[set]) -> bool:
    """Determine if a file should be searched."""
    ext = file_path.suffix.lower()
    if filter_extensions:
        return ext in filter_extensions
    return ext in DEFAULT_TEXT_EXTENSIONS


def _search_file(
    file_path: Path,
    pattern: re.Pattern,
    context_lines: int,
    max_results: int,
    current_count: int,
) -> List[Dict[str, Any]]:
    """Search a single file for pattern matches."""
    matches = []
    remaining = max_results - current_count

    if remaining <= 0:
        return matches

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except (IOError, PermissionError):
        return matches

    for i, line in enumerate(lines):
        if len(matches) >= remaining:
            break

        if pattern.search(line):
            # Get context lines
            start_ctx = max(0, i - context_lines)
            end_ctx = min(len(lines), i + context_lines + 1)

            context_before = [
                {"line_number": start_ctx + j + 1, "content": lines[start_ctx + j].rstrip("\n\r")}
                for j in range(i - start_ctx)
            ]
            context_after = [
                {"line_number": i + j + 2, "content": lines[i + j + 1].rstrip("\n\r")}
                for j in range(end_ctx - i - 1)
            ]

            matches.append({
                "file": str(file_path.resolve()),
                "line_number": i + 1,
                "match_line": line.rstrip("\n\r"),
                "context_before": context_before,
                "context_after": context_after,
            })

    return matches


async def main(**kwargs) -> Dict[str, Any]:
    """
    Search for a pattern in files within a directory.
    Returns dict with: pattern, matches, total_matches, truncated
    """
    pattern_str = kwargs.get("pattern")
    directory_path = kwargs.get("directory_path", ".")
    file_patterns = kwargs.get("file_patterns")
    case_sensitive = kwargs.get("case_sensitive", False)
    use_regex = kwargs.get("use_regex", False)
    context_lines = kwargs.get("context_lines", 1)
    max_results = kwargs.get("max_results", 50)
    recursive = kwargs.get("recursive", True)

    if not pattern_str:
        raise ValueError("pattern is required")

    # Validate directory
    dir_path = Path(directory_path)
    if not dir_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Compile the pattern
    flags = 0 if case_sensitive else re.IGNORECASE
    if use_regex:
        try:
            pattern = re.compile(pattern_str, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
    else:
        pattern = re.compile(re.escape(pattern_str), flags)

    # Prepare file extension filter
    filter_extensions = None
    if file_patterns:
        filter_extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in file_patterns
        }

    # Search files
    all_matches = []
    files_searched = 0

    if recursive:
        for root, _, files in os.walk(dir_path):
            for filename in files:
                file_path = Path(root) / filename
                if not _should_search_file(file_path, filter_extensions):
                    continue

                files_searched += 1
                file_matches = _search_file(
                    file_path, pattern, context_lines, max_results, len(all_matches)
                )
                all_matches.extend(file_matches)

                if len(all_matches) >= max_results:
                    break
            if len(all_matches) >= max_results:
                break
    else:
        for item in dir_path.iterdir():
            if not item.is_file():
                continue
            if not _should_search_file(item, filter_extensions):
                continue

            files_searched += 1
            file_matches = _search_file(
                item, pattern, context_lines, max_results, len(all_matches)
            )
            all_matches.extend(file_matches)

            if len(all_matches) >= max_results:
                break

    return {
        "pattern": pattern_str,
        "matches": all_matches,
        "total_matches": len(all_matches),
        "files_searched": files_searched,
        "truncated": len(all_matches) >= max_results,
    }
