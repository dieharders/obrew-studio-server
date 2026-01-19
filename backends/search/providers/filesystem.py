"""
FileSystemProvider - Search provider for local file system.

This provider implements the SearchProvider protocol for searching
local directories using file tools (file_scan, file_preview, file_read, file_parse).
"""

import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..base import SearchProvider, SearchItem


class FileSystemProvider(SearchProvider):
    """
    Search provider for local file system.

    Uses file tools to scan directories, preview files, and extract content.
    Implements directory whitelisting for security.
    """

    def __init__(
        self,
        app,
        allowed_directories: List[str],
        file_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the FileSystemProvider.

        Args:
            app: FastAPI application instance
            allowed_directories: List of directory paths the provider is allowed to access
            file_patterns: Optional list of file patterns to filter (e.g., ["*.pdf", "*.docx"])
        """
        self.app = app
        self.allowed_directories = [Path(d).resolve() for d in allowed_directories]
        self.file_patterns = file_patterns
        self._tool_cache: Dict[str, Any] = {}

    def _validate_path(self, path: str) -> bool:
        """
        Ensure a path is within the allowed directories.

        Args:
            path: Path to validate

        Returns:
            True if path is allowed, False otherwise
        """
        try:
            resolved = Path(path).resolve()
            return any(
                resolved == allowed or resolved.is_relative_to(allowed)
                for allowed in self.allowed_directories
            )
        except (ValueError, OSError):
            return False

    def _load_tool(self, tool_name: str):
        """
        Dynamically load a file tool module.

        Args:
            tool_name: Name of the tool to load

        Returns:
            The tool module
        """
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        try:
            module = importlib.import_module(f"tools.built_in_functions.{tool_name}")
            self._tool_cache[tool_name] = module
            return module
        except ImportError as e:
            raise ValueError(f"Failed to load tool '{tool_name}': {e}")

    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a file tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
        # Validate paths in kwargs
        for key in ["directory_path", "file_path"]:
            if key in kwargs and kwargs[key]:
                if not self._validate_path(kwargs[key]):
                    raise ValueError(
                        f"Access denied: '{kwargs[key]}' is outside allowed directories"
                    )

        tool = self._load_tool(tool_name)
        return await tool.main(**kwargs)

    async def discover(self, scope: str, **kwargs) -> List[SearchItem]:
        """
        Discover files in the given directory.

        Args:
            scope: Directory path to scan
            **kwargs: Additional arguments (query is ignored for file scan)

        Returns:
            List of SearchItem objects representing discovered files
        """
        # Validate directory
        if not self._validate_path(scope):
            raise ValueError(f"Access denied: '{scope}' is outside allowed directories")

        try:
            scan_result = await self._execute_tool(
                "file_scan",
                directory_path=scope,
                file_patterns=self.file_patterns,
                recursive=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to scan directory: {e}")

        if not scan_result:
            return []

        # Convert to SearchItem format (limit to first 50)
        items = []
        for idx, file_info in enumerate(scan_result[:50]):
            items.append(
                SearchItem(
                    id=file_info.get("path", ""),
                    name=file_info.get(
                        "filename", file_info.get("relative_path", "unknown")
                    ),
                    type="file",
                    preview=f"{file_info.get('type', 'file')} - {file_info.get('size', '')}",
                    metadata={
                        "relative_path": file_info.get("relative_path"),
                        "extension": file_info.get("extension"),
                        "size": file_info.get("size"),
                        "size_bytes": file_info.get("size_bytes"),
                        "modified": file_info.get("modified"),
                        "file_type": file_info.get("type"),
                    },
                    requires_extraction=False,
                )
            )

        return items

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given files.

        Args:
            items: List of file items to preview

        Returns:
            Same items with preview field populated
        """
        from core import common

        for item in items:
            try:
                preview_result = await self._execute_tool(
                    "file_preview",
                    file_path=item.id,
                    max_chars=500,
                    max_lines=20,
                )

                # Update preview with actual content
                preview_text = preview_result.get("preview", "")
                if isinstance(preview_text, dict):
                    preview_text = str(preview_text)

                item.preview = preview_text[:500] if preview_text else "[No preview available]"

                # Check if file requires parsing (binary documents)
                item.requires_extraction = preview_result.get("requires_parsing", False)
                item.metadata = item.metadata or {}
                item.metadata["requires_parsing"] = item.requires_extraction
                item.metadata["is_text"] = preview_result.get("is_text", True)

            except Exception as e:
                print(
                    f"{common.PRNT_API} [FileSystemProvider] Preview failed for {item.name}: {e}",
                    flush=True,
                )
                item.preview = f"[Preview failed: {e}]"

        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given files.

        Args:
            items: List of file items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        from core import common

        context = []

        for item in items:
            try:
                # Decide whether to parse or read based on metadata
                requires_parsing = (
                    item.metadata.get("requires_parsing", False)
                    if item.metadata
                    else False
                )

                if requires_parsing:
                    # Use file_parse for documents (PDF, DOCX, etc.)
                    result = await self._execute_tool(
                        "file_parse",
                        file_path=item.id,
                        output_format="text",
                    )
                    content = result.get("content", "")
                else:
                    # Use file_read for text files
                    result = await self._execute_tool(
                        "file_read",
                        file_path=item.id,
                    )
                    content = result.get("content", "")

                context.append(
                    {
                        "source": item.name,
                        "content": content[:5000],  # Limit context size
                    }
                )

            except Exception as e:
                print(
                    f"{common.PRNT_API} [FileSystemProvider] Extract failed for {item.name}: {e}",
                    flush=True,
                )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return other allowed directories that can be searched.

        Args:
            current_scope: The directory that was just searched

        Returns:
            List of other allowed directory paths
        """
        current_resolved = Path(current_scope).resolve()
        return [
            str(d)
            for d in self.allowed_directories
            if d != current_resolved and not current_resolved.is_relative_to(d)
        ]
