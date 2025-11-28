import os
import glob
import json
import uuid
from typing import Callable, List, Optional
from core import common, classes
from core.document import IndexNode
from chromadb import Collection, PersistentClient
from chromadb.api import ClientAPI
from chromadb.config import Settings


VECTOR_DB_FOLDER = "chromadb"
VECTOR_STORAGE_PATH = common.app_path(VECTOR_DB_FOLDER)


class Vector_Storage:
    """Handles document storage and retrieval using ChromaDB directly."""

    def __init__(
        self,
        app: classes.FastAPIApp,
        embed_fn: Callable[[str], List[float]] = None,
    ):
        self.app = app
        self.embed_fn = embed_fn
        self.db_client = self._get_vector_db_client()

    def _get_vector_db_client(self) -> ClientAPI:
        """Create a ChromaDB client singleton"""
        if not self.app.state.db_client:
            print(f"{common.PRNT_API} Connecting to vector store...")
            # Initialize storage path
            if not os.path.exists(VECTOR_STORAGE_PATH):
                os.makedirs(VECTOR_STORAGE_PATH)

            self.app.state.db_client = PersistentClient(
                path=VECTOR_STORAGE_PATH,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
        return self.app.state.db_client

    # @TODO Implement this for knowledge base actions when uploading documents, or delete.
    # This is a newer example of a simpler implementation, not used yet.
    def add_documents(
        self,
        docs: List[str],
        collection: Collection,
        embed_fn: Callable,
        metadata: Optional[List[dict]] = None,
    ):
        ids = [str(uuid.uuid4()) for _ in docs]
        embeddings = [embed_fn(doc) for doc in docs]
        metadatas = metadata if metadata else [None for _ in docs]
        collection.add(
            documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    def get_sources_from_ids(
        self, collection: Collection, source_ids: List[str]
    ) -> List[classes.SourceMetadata]:
        """Return source(s) given id(s) in a collection"""
        all_sources = self.get_collection_sources(collection)
        sources = []
        for s in all_sources:
            if s.get("id") in source_ids:
                sources.append(s)
        return sources

    def get_collection_sources(
        self, collection: Collection
    ) -> List[classes.SourceMetadata]:
        """Return the list of sources in this collection"""
        sources_json = collection.metadata.get("sources")
        if sources_json and type(sources_json) == str:
            return json.loads(sources_json)
        return sources_json

    def add_chunks_to_collection(
        self,
        collection: Collection,
        nodes: List[IndexNode],
    ) -> None:
        """Add chunks directly to ChromaDB collection."""
        if not nodes:
            return

        if not self.embed_fn:
            raise ValueError("embed_fn is required to add chunks to collection")

        ids = [node.id_ for node in nodes]
        documents = [node.text for node in nodes]
        embeddings = [self.embed_fn(node.text) for node in nodes]
        metadatas = [node.metadata for node in nodes]

        print(
            f"{common.PRNT_API} Adding {len(nodes)} chunks to collection...", flush=True
        )

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def update_collection_sources(
        self, collection: Collection, sources: List[classes.SourceMetadata], mode="add"
    ):
        """Add/remove or update Collection's metadata.sources list"""
        prev_sources = self.get_collection_sources(collection)
        new_sources = []

        if mode == "add":
            if prev_sources:
                prev_sources.extend(sources)
                new_sources = prev_sources
            else:
                new_sources = sources
            print(
                f"{common.PRNT_API} Added {len(new_sources)} new sources to collection."
            )
        elif mode == "delete":
            for i in sources:
                if i in prev_sources:
                    prev_sources.remove(i)
            new_sources = prev_sources
            print(
                f"{common.PRNT_API} Removed {len(new_sources)} sources from collection."
            )

        collection.metadata["sources"] = json.dumps(new_sources)
        collection.modify(metadata=collection.metadata)

    def list_collections(self, tenant="default") -> List[str]:
        """Returns all collection names in the specified db"""
        collections_list = self.db_client.list_collections()
        collection_names = []
        for coll in collections_list:
            collection_names.append(coll.name)
        return collection_names

    def get_all_collections(self, tenant="default") -> List[dict]:
        """Returns all collections and their metadata in the specified db"""
        collection_names = self.list_collections()
        collections = []
        for name in collection_names:
            collection = self.db_client.get_collection(name)
            sources = self.get_collection_sources(collection)
            collection.metadata["sources"] = sources
            collections.append(collection)
        return collections

    def get_collection(self, name: str, tenant="default") -> Optional[Collection]:
        """Returns a single collection and its metadata"""
        collection = self.db_client.get_collection(name) or None
        if collection:
            sources = self.get_collection_sources(collection)
            collection.metadata["sources"] = sources
            return collection
        raise Exception("No collection found.")

    def get_source_chunks(self, collection_name: str, source_id: str):
        """Returns all documents (chunks) associated with a source"""
        collection = self.db_client.get_collection(collection_name)
        doc_chunks = collection.get(where={"sourceId": source_id})

        chunks = []
        doc_chunk_ids = doc_chunks["ids"]
        for i, chunk_id in enumerate(doc_chunk_ids):
            chunk_text = doc_chunks["documents"][i]
            chunk_metadata = doc_chunks["metadatas"][i]
            chunk_metadata["_node_content"] = json.loads(
                chunk_metadata["_node_content"]
            )
            result = dict(
                id=chunk_id,
                text=chunk_text,
                metadata=chunk_metadata,
            )
            chunks.append(result)

        print(f"{common.PRNT_API} Returned {len(chunks)} chunks")
        return chunks

    def delete_chunks(
        self,
        collection: Collection,
        chunk_ids: List[str],
    ):
        """Delete chunk embeddings from ChromaDB collection."""
        if chunk_ids:
            collection.delete(ids=chunk_ids)
            print(f"{common.PRNT_API} Deleted {len(chunk_ids)} chunks from collection")

    def delete_sources(
        self,
        collection_name: str,
        sources: List[classes.SourceMetadata],
    ):
        """Delete all associated document chunks, metadata and files for given sources."""
        collection = self.get_collection(name=collection_name)
        # Delete each source chunk and parsed file
        for source in sources:
            chunk_ids = source.get("chunkIds")
            # Delete all chunks
            self.delete_chunks(
                collection=collection,
                chunk_ids=chunk_ids,
            )
            # Delete associated files
            self.delete_source_files(source)
        # Update collection metadata.sources to remove this source
        self.update_collection_sources(
            collection=collection,
            sources=sources,
            mode="delete",
        )

    def delete_source_files(self, source: classes.SourceMetadata):
        """Delete all files and references associated with embedded docs"""
        source_file_path = source.get("file_path")
        source_id = source.get("id")
        if os.path.exists(source_file_path):
            print(f"{common.PRNT_API} Remove file {source_id} from {source_file_path}")
            os.remove(source_file_path)

    def delete_all_vector_storage(self):
        """Remove all vector storage collections and folders"""
        if os.path.exists(VECTOR_STORAGE_PATH):
            try:
                files = glob.glob(f"{VECTOR_STORAGE_PATH}/*")
                for f in files:
                    os.remove(f)
                os.rmdir(VECTOR_STORAGE_PATH)
            except Exception as e:
                print(f"{common.PRNT_API} Failed to remove vector storage: {e}")
