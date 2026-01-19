"""File preview tool for quick file content summaries."""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field


# Supported text file extensions for quick preview
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".json", ".xml", ".yaml", ".yml",
    ".csv", ".tsv", ".log", ".ini", ".cfg", ".conf", ".properties",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".r", ".R", ".jl", ".lua", ".pl", ".pm",
    ".toml", ".env", ".gitignore", ".dockerfile",
}


class Params(BaseModel):
    """Get a quick preview of a file's content and metadata. Use this before deciding to fully parse a file."""

    file_path: str = Field(
        ...,
        description="The path to the file to preview.",
    )
    max_chars: int = Field(
        default=500,
        description="Maximum number of characters to include in the preview.",
    )
    max_lines: int = Field(
        default=20,
        description="Maximum number of lines to include in the preview.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "file_path": "/documents/report.txt",
                    "max_chars": 500,
                    "max_lines": 20,
                }
            ]
        }
    }


def _is_text_file(file_path: Path) -> bool:
    """Check if a file is likely a text file based on extension."""
    return file_path.suffix.lower() in TEXT_EXTENSIONS


def _is_binary(content: bytes) -> bool:
    """Check if content appears to be binary."""
    # Check for null bytes which indicate binary
    return b'\x00' in content[:1024]


async def main(**kwargs) -> Dict[str, Any]:
    """
    Get a preview of a file's content and metadata.
    Returns dict with: filename, path, type, size, modified, preview, preview_truncated
    """
    file_path_str = kwargs.get("file_path")
    max_chars = kwargs.get("max_chars", 500)
    max_lines = kwargs.get("max_lines", 20)

    if not file_path_str:
        raise ValueError("file_path is required")

    file_path = Path(file_path_str)

    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path_str}")
    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path_str}")

    # Get file metadata
    stat = file_path.stat()
    extension = file_path.suffix.lower()

    result = {
        "filename": file_path.name,
        "path": str(file_path.resolve()),
        "extension": extension,
        "size_bytes": stat.st_size,
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "preview": None,
        "preview_truncated": False,
        "is_text": False,
        "requires_parsing": False,
    }

    # Determine if file needs special parsing (PDF, DOCX, etc.)
    parsing_extensions = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt", ".rtf"}
    if extension in parsing_extensions:
        result["requires_parsing"] = True
        result["preview"] = f"[Binary document: {extension.upper()[1:]} file. Use file_parse tool for full content extraction.]"
        return result

    # Try to read as text
    if _is_text_file(file_path) or stat.st_size < 100_000:  # Try small files regardless
        try:
            # Read first chunk to check for binary
            with open(file_path, "rb") as f:
                initial_bytes = f.read(min(1024, stat.st_size))

            if _is_binary(initial_bytes):
                result["preview"] = f"[Binary file: {extension or 'unknown format'}]"
                return result

            # Read as text
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = []
                total_chars = 0

                for i, line in enumerate(f):
                    if i >= max_lines or total_chars >= max_chars:
                        result["preview_truncated"] = True
                        break
                    lines.append(line.rstrip("\n\r"))
                    total_chars += len(line)

                result["preview"] = "\n".join(lines)
                result["is_text"] = True

                # Check if we hit the char limit mid-line
                if total_chars > max_chars:
                    result["preview"] = result["preview"][:max_chars]
                    result["preview_truncated"] = True

        except (UnicodeDecodeError, IOError):
            result["preview"] = f"[Unable to read file: encoding error or access denied]"
    else:
        result["preview"] = f"[Large file ({stat.st_size} bytes). Use file_read for specific sections.]"
        result["preview_truncated"] = True

    return result
