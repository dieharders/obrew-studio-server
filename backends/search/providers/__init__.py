"""
Search providers for the unified search architecture.

This package contains provider implementations for different context types:
- FileSystemProvider: Searches local file system
- VectorProvider: Searches vector/embedding collections
- WebProvider: Searches the web using DuckDuckGo
- StructuredProvider: Searches client-provided structured data
"""

from .filesystem import FileSystemProvider
from .vector import VectorProvider
from .web import WebProvider
from .structured import StructuredProvider

__all__ = ["FileSystemProvider", "VectorProvider", "WebProvider", "StructuredProvider"]
