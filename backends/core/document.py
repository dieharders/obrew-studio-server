"""
Replacement data classes for llama-index Document, TextNode, and IndexNode.
These simple dataclasses provide the same interface without the heavy dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class Document:
    """Simple document with text and metadata."""

    id_: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    excluded_llm_metadata_keys: List[str] = field(default_factory=list)
    excluded_embed_metadata_keys: List[str] = field(default_factory=list)

    def get_content(self) -> str:
        """Return the document text content."""
        return self.text


@dataclass
class TextNode:
    """A text chunk/node from document parsing."""

    text: str
    id_: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_content(self) -> str:
        """Return the node text content."""
        return self.text


@dataclass
class IndexNode:
    """A chunk node for vector indexing."""

    id_: str
    text: str
    index_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    excluded_llm_metadata_keys: List[str] = field(default_factory=list)
    excluded_embed_metadata_keys: List[str] = field(default_factory=list)
    metadata_seperator: str = "::"
    metadata_template: str = "{key}=>{value}"
    text_template: str = "Metadata: {metadata_str}\n-----\nContent: {content}"

    @property
    def node_id(self) -> str:
        """Alias for id_ for compatibility."""
        return self.id_

    def get_content(self) -> str:
        """Return the node text content."""
        return self.text
