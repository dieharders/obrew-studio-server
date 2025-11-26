# Plan: Remove llama-index Dependency

## Goal
Remove `llama-index` and related packages to eliminate the NLTK dependency that causes SSL certificate errors in bundled macOS builds.

## Current State

### Files Using llama-index (7 files)
1. `backends/embeddings/embedder.py` - VectorStoreIndex, Document, HuggingFaceEmbedding, callbacks
2. `backends/embeddings/vector_storage.py` - IndexNode, ChromaVectorStore, VectorStoreIndex, StorageContext
3. `backends/embeddings/file_loaders.py` - SimpleDirectoryReader, Document, file readers (PyMuPDFReader, DocxReader, etc.)
4. `backends/embeddings/chunking.py` - Document, IndexNode, TextNode
5. `backends/embeddings/text_splitters.py` - SimpleDirectoryReader, node parsers (SentenceSplitter, MarkdownNodeParser)
6. `backends/embeddings/query.py` - VectorStoreIndex, PromptTemplate, ResponseMode
7. `backends/embeddings/evaluation.py` - FaithfulnessEvaluator
8. `backends/embeddings/gguf_embedder.py` - BaseEmbedding base class
9. `backends/inference/helpers.py` - Document (from llama_index_client)

### Good News
- `backends/retrieval/rag.py` already uses ChromaDB directly without llama-index
- You already have `sentence-transformers` which can replace `llama-index-embeddings-huggingface`
- You already have `PyMuPDF` which can replace `PyMuPDFReader`
- You already have `chromadb` which can be used directly

## Implementation Plan

### Phase 1: Create Replacement Data Classes

**File: `backends/core/document.py` (new)**

Create simple dataclasses to replace llama-index Document, TextNode, IndexNode:

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class Document:
    """Simple document with text and metadata."""
    id_: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    excluded_llm_metadata_keys: List[str] = field(default_factory=list)
    excluded_embed_metadata_keys: List[str] = field(default_factory=list)

    def get_content(self) -> str:
        return self.text

@dataclass
class TextNode:
    """A text chunk/node."""
    id_: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class IndexNode:
    """A chunk node for indexing."""
    id_: str
    text: str
    index_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    node_id: str = ""
    excluded_llm_metadata_keys: List[str] = field(default_factory=list)
    excluded_embed_metadata_keys: List[str] = field(default_factory=list)
    metadata_seperator: str = "::"
    metadata_template: str = "{key}=>{value}"
    text_template: str = "Metadata: {metadata_str}\n-----\nContent: {content}"

    def __post_init__(self):
        self.node_id = self.id_
```

### Phase 2: Replace Embeddings

**File: `backends/embeddings/embedder.py`**

Replace `HuggingFaceEmbedding` with direct `sentence-transformers` usage:

```python
from sentence_transformers import SentenceTransformer

class DirectEmbedder:
    def __init__(self, model_name: str, cache_folder: str):
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)

    def get_text_embedding(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()
```

### Phase 3: Replace File Readers

**File: `backends/embeddings/file_loaders.py`**

Replace llama-index readers with native libraries:

| llama-index Reader | Replacement |
|-------------------|-------------|
| `SimpleDirectoryReader` | Direct file reading with `open()` |
| `PyMuPDFReader` | Direct `fitz` (PyMuPDF) - already have it |
| `DocxReader` | `python-docx` library |
| `CSVReader` | Standard `csv` module |
| `RTFReader` | `striprtf` library |
| `XMLReader` | Standard `xml.etree.ElementTree` |
| `PptxReader` | `python-pptx` library |
| `ImageReader` | Return empty (no text extraction) or use pytesseract |
| `VideoAudioReader` | Not used in practice, can stub |

### Phase 4: Replace Text Splitters

**File: `backends/embeddings/text_splitters.py`**

Replace `SentenceSplitter` with regex-based splitting (you already have this pattern):

```python
import re
from typing import List
from core.document import TextNode

class SimpleSentenceSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0, paragraph_separator: str = "\n\n"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.paragraph_separator = paragraph_separator

    def get_nodes_from_documents(self, documents: List, show_progress: bool = False) -> List[TextNode]:
        nodes = []
        for doc in documents:
            text = doc.text
            # Split by paragraph separator first
            paragraphs = text.split(self.paragraph_separator)

            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) <= self.chunk_size:
                    current_chunk += para + self.paragraph_separator
                else:
                    if current_chunk:
                        nodes.append(TextNode(id_=str(len(nodes)), text=current_chunk.strip()))
                    current_chunk = para + self.paragraph_separator

            if current_chunk:
                nodes.append(TextNode(id_=str(len(nodes)), text=current_chunk.strip()))

        return nodes
```

### Phase 5: Replace Vector Storage

**File: `backends/embeddings/vector_storage.py`**

Remove the llama-index wrapper and use ChromaDB directly. The key change is in `add_chunks_to_collection`:

```python
def add_chunks_to_collection(
    self,
    collection: Collection,
    nodes: List[IndexNode],
    embed_fn: Callable[[str], List[float]],
) -> None:
    """Add chunks directly to ChromaDB collection."""
    if not nodes:
        return

    ids = [node.id_ for node in nodes]
    documents = [node.text for node in nodes]
    embeddings = [embed_fn(node.text) for node in nodes]
    metadatas = [node.metadata for node in nodes]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
```

### Phase 6: Update Query/RAG

**File: `backends/embeddings/query.py`**

The existing `backends/retrieval/rag.py` already shows how to query ChromaDB directly. Migrate to use that pattern:

```python
def query_collection(
    collection: Collection,
    query_embedding: List[float],
    top_k: int = 5,
) -> List[dict]:
    """Query ChromaDB directly."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results
```

### Phase 7: Update GGUFEmbedder

**File: `backends/embeddings/gguf_embedder.py`**

Remove inheritance from `BaseEmbedding`, make it a simple class:

```python
class GGUFEmbedder:
    """Handle GGUF embedding models using llama-cli binary."""

    def __init__(self, app, model_path: str, embed_model: str):
        self.model_name = embed_model
        # ... rest of init

    def get_text_embedding(self, text: str) -> List[float]:
        return self.embed_text(text, normalize=True)
```

### Phase 8: Remove evaluation.py

The `FaithfulnessEvaluator` requires an LLM and is marked as needing refactor. This file can be:
- Removed entirely (if not used)
- Stubbed out for future implementation

### Phase 9: Update Requirements

**File: `requirements.txt`**

Remove:
```
llama-index==0.10.29
llama-index-vector-stores-chroma==0.1.8
llama-index-embeddings-huggingface==0.2.0
llama-parse==0.4.9
```

Add (if not already present):
```
python-docx==1.1.0
python-pptx==0.6.23
striprtf==0.0.26
```

Keep:
```
chromadb==0.5.0
sentence-transformers==2.6.1
PyMuPDF==1.25.5
```

## File Changes Summary

| File | Action |
|------|--------|
| `backends/core/document.py` | CREATE - New dataclasses |
| `backends/embeddings/embedder.py` | MODIFY - Use DirectEmbedder |
| `backends/embeddings/vector_storage.py` | MODIFY - Use ChromaDB directly |
| `backends/embeddings/file_loaders.py` | MODIFY - Use native libraries |
| `backends/embeddings/chunking.py` | MODIFY - Use new dataclasses |
| `backends/embeddings/text_splitters.py` | MODIFY - Use simple splitter |
| `backends/embeddings/query.py` | MODIFY - Use direct ChromaDB query |
| `backends/embeddings/evaluation.py` | REMOVE or STUB |
| `backends/embeddings/gguf_embedder.py` | MODIFY - Remove BaseEmbedding |
| `backends/inference/helpers.py` | MODIFY - Use new Document class |
| `requirements.txt` | MODIFY - Remove llama-index packages |
| `Obrew-Studio.spec` | MODIFY - Remove llama-index data paths |

## Benefits

1. **Eliminates NLTK** - No more SSL certificate errors in bundled app
2. **Smaller bundle size** - llama-index pulls in many dependencies
3. **Faster startup** - Less initialization overhead
4. **Simpler codebase** - Direct API usage is easier to debug
5. **No runtime downloads** - All data bundled at build time

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Sentence splitting quality | The regex approach is simple but effective; can enhance later |
| Missing edge cases in file readers | Test with variety of files before release |
| Breaking existing embeddings | Embeddings are stored in ChromaDB, format unchanged |

## Testing Plan

1. Create embeddings from various file types (PDF, DOCX, TXT, etc.)
2. Query existing collections
3. Build PyInstaller bundle
4. Run on clean macOS system
5. Verify no NLTK/SSL errors
