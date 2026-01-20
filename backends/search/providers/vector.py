"""
VectorProvider - Search provider for vector/embedding collections.

This provider implements the SearchProvider protocol for searching
ChromaDB vector collections using semantic similarity.

Supports two modes:
- Discovery mode (no collections specified): Lists all collections with metadata for LLM selection
- Search mode (collections specified): Semantic search within specified collections
"""

import asyncio
from typing import List, Dict, Optional

from ..harness import SearchProvider, SearchItem


def _create_vision_embed_fn(vision_embedder):
    """
    Create an async embed function for the vision embedder.
    Defined outside class to avoid closure issues.
    """

    async def vision_embed_fn(text: str) -> List[float]:
        return await vision_embedder.embed_query_text(text, auto_unload=False)

    return vision_embed_fn


class VectorProvider(SearchProvider):
    """
    Search provider for vector/embedding collections.

    Uses ChromaDB for vector storage and retrieval with semantic similarity search.
    Supports context expansion via original_text metadata (sentence window retrieval).

    When no collections are specified, operates in discovery mode - listing all available
    collections with their metadata so the LLM can select which to search.
    """

    def __init__(self, app, collections: Optional[List[str]] = None, top_k: int = 50):
        """
        Initialize the VectorProvider.

        Args:
            app: FastAPI application instance
            collections: Optional list of collection names to search.
                        If None/empty, operates in discovery mode.
            top_k: Maximum number of chunks to retrieve per collection
        """
        self.app = app
        self.collections = collections or []
        self.top_k = top_k
        self._embedder = None
        self._vision_embedder = None
        self._vector_storage = None
        self._current_query = None  # Store query for use in extract()
        self._searched_collections = []  # Track which collections have been searched

    def _get_vector_storage(self):
        """Lazy-load the vector storage."""
        if self._vector_storage is None:
            from embeddings.vector_storage import Vector_Storage

            self._vector_storage = Vector_Storage(app=self.app)
        return self._vector_storage

    def _get_embed_fn(self, collection):
        """
        Get appropriate embed function based on collection type.

        Returns a text or vision embed function depending on collection metadata.
        """
        collection_type = collection.metadata.get("type", "")

        if collection_type == "image_embeddings":
            # Use vision embedder for image collections
            if self._vision_embedder is None:
                from vision.image_embedder import ImageEmbedder

                self._vision_embedder = ImageEmbedder(self.app)
            return _create_vision_embed_fn(self._vision_embedder)
        else:
            # Use text embedder for text collections
            if self._embedder is None:
                from embeddings.embedder import Embedder

                embed_model = collection.metadata.get("embedding_model")
                self._embedder = Embedder(app=self.app, embed_model=embed_model)
            return self._embedder.embed_text

    async def _get_embedding(self, text: str, embed_fn) -> List[float]:
        """
        Get embedding for text, handling both sync and async embed functions.

        Args:
            text: Text to embed
            embed_fn: Embed function (sync or async)

        Returns:
            Embedding vector
        """
        result = embed_fn(text)
        # If embed_fn returned a coroutine, await it
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _query_collection(
        self, collection_name: str, query: str
    ) -> List[Dict[str, str]]:
        """
        Query a collection and return context dicts.

        Args:
            collection_name: Name of the collection to query
            query: Search query text

        Returns:
            List of context dicts with 'source' and 'content' keys
        """
        from core import common

        vector_storage = self._get_vector_storage()
        collection = vector_storage.get_collection(collection_name)

        # Get embed function for this collection
        embed_fn = self._get_embed_fn(collection)
        query_embedding = await self._get_embedding(query, embed_fn)

        # Validate embedding dimensions
        query_dim = len(query_embedding)
        expected_dim = collection.metadata.get("embedding_dim")
        if expected_dim and expected_dim != query_dim:
            print(
                f"{common.PRNT_API} [VectorProvider] Skipping collection '{collection_name}': "
                f"dimension mismatch (expected {expected_dim}, got {query_dim})",
                flush=True,
            )
            return []

        # Query the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # Convert to context format
        context = []
        for doc, meta in zip(documents, metadatas):
            original_text = meta.get("original_text") if meta else None
            source_id = meta.get("sourceId", "unknown") if meta else "unknown"

            if original_text and original_text != doc:
                content = f"[Matched]: {doc}\n[Context]: {original_text}"
            else:
                content = doc or ""

            context.append(
                {
                    "source": f"[{collection_name}] {source_id}",
                    "content": content[:5000],
                }
            )

        print(
            f"{common.PRNT_API} [VectorProvider] Queried collection '{collection_name}': {len(context)} results",
            flush=True,
        )

        return context

    async def _search_collection(
        self, collection_name: str, query: str
    ) -> List[SearchItem]:
        """
        Search a single collection and return SearchItems.

        Args:
            collection_name: Name of the collection to search
            query: Search query text

        Returns:
            List of SearchItem objects (chunks)
        """
        from core import common

        vector_storage = self._get_vector_storage()

        try:
            collection = vector_storage.get_collection(collection_name)

            # Get embed function for this collection (text or vision based on type)
            embed_fn = self._get_embed_fn(collection)

            # Create query embedding
            query_embedding = await self._get_embedding(query, embed_fn)

            # Validate embedding dimensions
            query_dim = len(query_embedding)
            expected_dim = collection.metadata.get("embedding_dim")
            if expected_dim and expected_dim != query_dim:
                print(
                    f"{common.PRNT_API} [VectorProvider] Skipping collection '{collection_name}': "
                    f"dimension mismatch (expected {expected_dim}, got {query_dim})",
                    flush=True,
                )
                return []

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
                            "collection": collection_name,
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
                f"{common.PRNT_API} [VectorProvider] Found {len(items)} chunks in '{collection_name}'",
                flush=True,
            )

            return items

        except Exception as e:
            print(
                f"{common.PRNT_API} [VectorProvider] Error searching '{collection_name}': {e}",
                flush=True,
            )
            return []

    async def discover(self, scope: Optional[str] = None, **kwargs) -> List[SearchItem]:
        """
        Discover items based on configured collections.

        Two modes:
        - Discovery mode (no collections configured): List all collections with metadata
        - Search mode (collections configured): Semantic search within those collections

        Args:
            scope: Ignored - uses self.collections instead
            **kwargs: Additional arguments (must include 'query')

        Returns:
            List of SearchItem objects (collections or chunks)
        """
        from core import common

        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query is required for vector search")

        # Store query for use in extract()
        self._current_query = query

        vector_storage = self._get_vector_storage()

        if not self.collections:
            # Discovery mode: list all collections with metadata
            print(
                f"{common.PRNT_API} [VectorProvider] Discovery mode: listing all collections",
                flush=True,
            )

            all_collections = vector_storage.get_all_collections()

            items = []
            for coll in all_collections:
                metadata = coll.get("metadata", {})
                sources = metadata.get("sources", [])
                source_count = len(sources) if isinstance(sources, list) else 0

                # Build preview from collection metadata
                description = metadata.get("description", "No description")
                tags = metadata.get("tags", "")
                coll_type = metadata.get("type", "text")

                preview = f"{description}"
                if tags:
                    preview += f" | Tags: {tags}"
                preview += f" | Type: {coll_type} | Sources: {source_count}"

                items.append(
                    SearchItem(
                        id=coll["name"],
                        name=coll["name"],
                        type="collection",
                        preview=preview,
                        metadata={
                            "embedding_model": metadata.get("embedding_model"),
                            "embedding_dim": metadata.get("embedding_dim"),
                            "type": coll_type,
                            "source_count": source_count,
                            "description": description,
                            "tags": tags,
                        },
                        requires_extraction=True,  # Need to query the collection
                    )
                )

            print(
                f"{common.PRNT_API} [VectorProvider] Found {len(items)} collections",
                flush=True,
            )

            return items

        else:
            # Search mode: semantic search within specified collections
            print(
                f"{common.PRNT_API} [VectorProvider] Search mode: querying {len(self.collections)} collection(s)",
                flush=True,
            )

            all_items = []
            for collection_name in self.collections:
                self._searched_collections.append(collection_name)
                items = await self._search_collection(collection_name, query)
                all_items.extend(items)

            # Sort by similarity (highest first)
            all_items.sort(
                key=lambda x: x.metadata.get("similarity", 0) if x.metadata else 0,
                reverse=True,
            )

            print(
                f"{common.PRNT_API} [VectorProvider] Found {len(all_items)} total chunks",
                flush=True,
            )

            return all_items

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """
        Get preview content for the given items.

        For collections: preview is already populated from metadata.
        For chunks: preview is already populated from discover phase.

        Args:
            items: List of items to preview

        Returns:
            Same items (preview already populated)
        """
        return items

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """
        Extract full content from the given items.

        For collections: Query them with semantic search.
        For chunks: Use original_text (sentence window context) if available.

        Args:
            items: List of items to extract content from

        Returns:
            List of dicts with 'source' and 'content' keys
        """
        context = []

        for item in items:
            if item.type == "collection":
                # Query this collection
                collection_context = await self._query_collection(
                    item.id, self._current_query
                )
                context.extend(collection_context)
                self._searched_collections.append(item.id)

            else:
                # Existing chunk extraction logic
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
        Return other collections that can be searched.

        Args:
            current_scope: The collection that was just searched

        Returns:
            List of other collection names not yet searched
        """
        all_collections = self._get_vector_storage().list_collections()
        return [
            c
            for c in all_collections
            if c != current_scope and c not in self._searched_collections
        ]

    async def close(self):
        """
        Clean up resources, especially vision embedder if used.

        Should be called when done with the provider to ensure
        proper cleanup of GPU/memory resources.
        """
        if self._vision_embedder is not None:
            await self._vision_embedder.unload()
            self._vision_embedder = None
