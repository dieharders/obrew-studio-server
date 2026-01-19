"""
VectorProvider - Search provider for vector/embedding collections.

This provider implements the SearchProvider protocol for searching
ChromaDB vector collections using semantic similarity.
"""

import asyncio
from typing import List, Dict

from ..base import SearchProvider, SearchItem


# @TODO This needs to know what embedder and vector length to use (look to RAG.py).
class VectorProvider(SearchProvider):
    """
    Search provider for vector/embedding collections.

    Uses ChromaDB for vector storage and retrieval with semantic similarity search.
    Supports context expansion via original_text metadata (sentence window retrieval).
    """

    def __init__(
        self,
        app,
        allowed_collections: List[str],
        top_k: int = 50,
    ):
        """
        Initialize the VectorProvider.

        Args:
            app: FastAPI application instance
            allowed_collections: List of collection names the provider is allowed to access
            top_k: Maximum number of chunks to retrieve per collection
        """
        self.app = app
        self.allowed_collections = allowed_collections
        self.top_k = top_k
        self._embedder = None
        self._vector_storage = None

    def _get_vector_storage(self):
        """Lazy-load the vector storage."""
        if self._vector_storage is None:
            from embeddings.vector_storage import Vector_Storage

            self._vector_storage = Vector_Storage(app=self.app)
        return self._vector_storage

    def _get_embedder(self, collection):
        """
        Get or create an embedder for the collection.

        Uses the embedding model specified in the collection metadata.
        """
        if self._embedder is None:
            from embeddings.embedder import Embedder

            # Get the embedding model from collection metadata
            embed_model = collection.metadata.get("embedding_model")
            self._embedder = Embedder(app=self.app, embed_model=embed_model)

        return self._embedder

    def _validate_collection(self, collection_name: str) -> bool:
        """
        Ensure a collection is in the allowed list.

        Args:
            collection_name: Name of the collection to validate

        Returns:
            True if collection is allowed, False otherwise
        """
        return collection_name in self.allowed_collections

    async def _get_embedding(self, text: str, embedder) -> List[float]:
        """
        Get embedding for text, handling both sync and async embedders.

        Args:
            text: Text to embed
            embedder: Embedder instance

        Returns:
            Embedding vector
        """
        result = embedder.embed_model.embed_text(text)
        # If embed_text returned a coroutine, await it
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def discover(self, scope: str, **kwargs) -> List[SearchItem]:
        """
        Discover chunks in the given collection via semantic search.

        Args:
            scope: Collection name to search
            **kwargs: Additional arguments (must include 'query' for semantic search)

        Returns:
            List of SearchItem objects representing discovered chunks
        """
        from core import common

        # Validate collection
        if not self._validate_collection(scope):
            raise ValueError(
                f"Access denied: collection '{scope}' is not in allowed list"
            )

        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for vector search")

        try:
            # Get vector storage and collection
            vector_storage = self._get_vector_storage()
            collection = vector_storage.get_collection(scope)

            # Get embedder for this collection
            embedder = self._get_embedder(collection)

            # Create query embedding
            query_embedding = await self._get_embedding(query, embedder)

            # Validate embedding dimensions
            query_dim = len(query_embedding)
            expected_dim = collection.metadata.get("embedding_dim")
            if expected_dim and expected_dim != query_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: collection has {expected_dim} "
                    f"dimensions but query embedding has {query_dim}."
                )

            # Query the collection
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=self.top_k,
            )

            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # Convert to SearchItem format
            items = []
            for i, (doc, meta, chunk_id, distance) in enumerate(
                zip(documents, metadatas, ids, distances)
            ):
                # Convert distance to similarity score
                similarity = 1 - distance if distance else 0

                items.append(
                    SearchItem(
                        id=chunk_id,
                        name=f"chunk_{i}",
                        type="chunk",
                        preview=doc[:300] if doc else "",
                        metadata={
                            "full_text": doc,
                            "original_text": (
                                meta.get("original_text") if meta else None
                            ),
                            "source_id": meta.get("sourceId") if meta else None,
                            "similarity": similarity,
                            "collection": scope,
                            **({k: v for k, v in meta.items()} if meta else {}),
                        },
                        requires_extraction=bool(
                            meta
                            and meta.get("original_text")
                            and meta.get("original_text") != doc
                        ),
                    )
                )

            print(
                f"{common.PRNT_API} [VectorProvider] Found {len(items)} chunks in collection '{scope}'",
                flush=True,
            )

            return items

        except Exception as e:
            raise ValueError(f"Failed to search collection '{scope}': {e}")

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given chunks.

        For vector search, chunks already have preview from discover phase.

        Args:
            items: List of chunk items to preview

        Returns:
            Same items (preview already populated)
        """
        # Chunks already have preview from the discover phase
        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given chunks.

        Uses original_text (sentence window context) if available,
        otherwise uses the chunk text.

        Args:
            items: List of chunk items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        context = []

        for item in items:
            metadata = item.metadata or {}

            # Get expanded context (original_text) if available
            original_text = metadata.get("original_text")
            full_text = metadata.get("full_text", item.preview)

            if original_text and original_text != full_text:
                # Include both matched chunk and expanded context
                content = f"[Matched]: {full_text}\n[Context]: {original_text}"
            else:
                content = full_text or item.preview or ""

            # Build source citation
            source_id = metadata.get("source_id", "unknown")
            collection = metadata.get("collection", "unknown")
            similarity = metadata.get("similarity", 0)

            context.append(
                {
                    "source": f"[{collection}] {source_id} (similarity: {similarity:.2f})",
                    "content": content[:5000],  # Limit context size
                }
            )

        return context

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """
        Return other allowed collections that can be searched.

        Args:
            current_scope: The collection that was just searched

        Returns:
            List of other allowed collection names
        """
        return [c for c in self.allowed_collections if c != current_scope]
