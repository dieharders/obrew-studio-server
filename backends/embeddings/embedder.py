import os
from datetime import datetime, timezone
from typing import List, TYPE_CHECKING
from fastapi import BackgroundTasks, UploadFile
from core import common
from core.classes import EmbedDocumentRequest, FastAPIApp
from core.document import Document
from .file_parsers import copy_file_to_disk, create_parsed_id, process_documents
from .text_splitters import (
    markdown_heading_split,
    markdown_document_split,
    recursive_character_split,
    sentence_window_split,
    token_split,
)
from .chunking import chunks_from_documents, create_source_record
from .file_loaders import documents_from_sources
from .gguf_embedder import GGUFEmbedder

if TYPE_CHECKING:
    from embeddings.vector_storage import Vector_Storage


# GGUF embedding models only
embedding_model_names = dict(
    nomic_v1_5="nomic-ai/nomic-embed-text-v1.5-GGUF",
)

DEFAULT_EMBEDDING_MODEL_NAME = embedding_model_names["nomic_v1_5"]
EMBEDDING_MODEL_CACHE_PATH = common.app_path("embed_models")
EMBEDDING_MODELS_CACHE_DIR = common.app_path(common.EMBEDDING_MODELS_CACHE_DIR)
CHUNKING_STRATEGIES = {
    "MARKDOWN_HEADING_SPLIT": markdown_heading_split,
    "MARKDOWN_DOCUMENT_SPLIT": markdown_document_split,
    "RECURSIVE_CHARACTER_SPLIT": recursive_character_split,
    "SENTENCE_WINDOW_SPLIT": sentence_window_split,
    "TOKEN_SPLIT": token_split,
}


def is_gguf_model(model_name: str) -> bool:
    """Check if a model identifier refers to a GGUF model."""
    if not model_name:
        return False
    # GGUF models will have repo names with "GGUF" in them
    return "GGUF" in model_name.upper() or model_name.lower().endswith(".gguf")


# How to use local embeddings: https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
# All models: https://huggingface.co/models?library=sentence-transformers
# Rankings: https://huggingface.co/spaces/mteb/leaderboard
class Embedder:
    """Handle vector embeddings."""

    def __init__(
        self,
        app: FastAPIApp,
        embed_model: str = None,
        cache_path: str = None,
    ):
        self.app = app
        self.cache = cache_path or EMBEDDING_MODEL_CACHE_PATH
        self.embed_model_name = embed_model or DEFAULT_EMBEDDING_MODEL_NAME
        self.is_gguf = is_gguf_model(self.embed_model_name)

        if not self.is_gguf:
            raise ValueError(
                f"Only GGUF embedding models are supported. "
                f"Model '{self.embed_model_name}' does not appear to be a GGUF model. "
                f"Please use a model with 'GGUF' in the name or ending in '.gguf'."
            )

        # For GGUF models, we need to find the model file path
        model_path = self._find_gguf_model_path()
        self.embed_model = GGUFEmbedder(
            app=app,
            model_path=model_path,
            embed_model=self.embed_model_name,
        )
        print(
            f"{common.PRNT_EMBED} Using GGUF embedder with model: {self.embed_model_name}",
            flush=True,
        )

    def _find_gguf_model_path(self) -> str:
        """Find the path to a GGUF model file."""
        # First check if embed_model_name is already a path
        if os.path.exists(self.embed_model_name) and self.embed_model_name.endswith(
            ".gguf"
        ):
            return self.embed_model_name

        # Look for GGUF files in the embedding models cache
        # The repo format is usually "owner/repo-name-GGUF"
        # Huggingface cache format: models--owner--repo-name-GGUF/snapshots/hash/*.gguf
        repo_slug = self.embed_model_name.replace("/", "--")
        repo_path = os.path.join(EMBEDDING_MODELS_CACHE_DIR, f"models--{repo_slug}")

        if os.path.exists(repo_path):
            # Look for .gguf files in snapshots subdirectories
            snapshots_path = os.path.join(repo_path, "snapshots")
            if os.path.exists(snapshots_path):
                for snapshot_dir in os.listdir(snapshots_path):
                    snapshot_full_path = os.path.join(snapshots_path, snapshot_dir)
                    if os.path.isdir(snapshot_full_path):
                        # Find all .gguf files in this snapshot
                        for file in os.listdir(snapshot_full_path):
                            if file.endswith(".gguf"):
                                gguf_path = os.path.join(snapshot_full_path, file)
                                print(
                                    f"{common.PRNT_EMBED} Found GGUF model at: {gguf_path}",
                                    flush=True,
                                )
                                return gguf_path

        raise FileNotFoundError(
            f"Could not find GGUF model file for {self.embed_model_name}. "
            f"Please ensure the model is downloaded to {EMBEDDING_MODELS_CACHE_DIR}"
        )

    # Create embeddings for one file (input) at a time
    # We create one source per upload
    def _create_new_embedding(
        self,
        vector_storage: "Vector_Storage",
        nodes: List[Document],
        chunk_size,
        chunk_overlap,
        chunk_strategy,
        collection_name,
    ):
        try:
            print(f"{common.PRNT_EMBED} Creating embeddings...", flush=True)
            # Loop through each file
            for node in nodes:
                # Create source document records for Collection metadata
                source_record = create_source_record(document=node)
                # Split document texts
                text_splitter = CHUNKING_STRATEGIES[chunk_strategy]
                splitter = text_splitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                # Build chunks
                print(f"{common.PRNT_EMBED} Chunking text...", flush=True)
                parsed_nodes = splitter.get_nodes_from_documents(
                    documents=[node],  # pass list of files (in Document format)
                    show_progress=True,
                )
                [chunks_ids, chunk_nodes] = chunks_from_documents(
                    source_record=source_record,
                    parsed_nodes=parsed_nodes,  # pass in text nodes to return as chunks
                    documents=[node],
                )
                # Record the ids of each chunk
                source_record["chunkIds"] = chunks_ids
                # Create/load a vector index, add chunks and persist to storage
                print(f"{common.PRNT_EMBED} Adding chunks to collection...", flush=True)

                collection = vector_storage.get_collection(name=collection_name)
                if collection:
                    vector_storage.add_chunks_to_collection(
                        collection=collection,
                        nodes=chunk_nodes,
                    )
                    # Add/update `collection.metadata.sources` list
                    print(
                        f"{common.PRNT_EMBED} Updating collection metadata...",
                        flush=True,
                    )
                    vector_storage.update_collection_sources(
                        collection=collection,
                        sources=[source_record],
                        mode="add",
                    )
            print(f"{common.PRNT_EMBED} Embedding succeeded!", flush=True)
        except (Exception, KeyError) as e:
            msg = f"Embedding failed: {e}"
            print(f"{common.PRNT_EMBED} {msg}", flush=True)

    def embed_text(self, text: str) -> List[float]:
        """Create vector embeddings from query text."""
        # Use the already initialized embed_model
        return self.embed_model.get_text_embedding(text)
        # Or
        # embedding_model = SentenceTransformer(
        #     self.embed_model_name, cache_folder=self.cache
        # )
        # return embedding_model.encode(text, normalize_embeddings=True).tolist()

    async def modify_document(
        self,
        vector_storage: "Vector_Storage",
        form: EmbedDocumentRequest,
        file: UploadFile,
        background_tasks: BackgroundTasks,
        is_update: bool = False,
    ):
        document_name = form.documentName
        prev_document_id = form.documentId
        source_name = form.documentName
        collection_name = form.collectionName
        description = form.description
        tags = common.parse_valid_tags(form.tags)
        url_path = form.urlPath
        local_file_path = form.filePath
        text_input = form.textInput
        chunk_size = form.chunkSize or 300
        chunk_overlap = form.chunkOverlap or 0
        chunk_strategy = form.chunkStrategy or list(CHUNKING_STRATEGIES.keys())[0]
        parsing_method = form.parsingMethod
        new_document_id = create_parsed_id(collection_name=collection_name)
        source_id = new_document_id
        if is_update:
            source_id = prev_document_id
        # Verify input values
        if (
            file == None  # file from client
            and url_path == ""  # file on web
            and text_input == ""  # text input from client
            and local_file_path == ""  # file on server disk
        ):
            raise Exception("Please supply a file upload, file path, url or text.")
        if not collection_name or collection_name == "undefined" or not source_name:
            raise Exception("Please supply a collection name and/or memory name.")
        if is_update and not prev_document_id:
            raise Exception("Please supply a document id.")
        if not document_name:
            raise Exception("Please supply a document name.")
        if not source_id:
            raise Exception("Server error, id misconfigured.")
        if not common.check_valid_id(source_name):
            raise Exception(
                "Invalid memory name. No '--', uppercase, spaces or special chars allowed."
            )
        if tags == None:
            raise Exception("Invalid value for 'tags' input.")
        # If updating, Remove specified source(s) from database
        if is_update:
            collection = vector_storage.get_collection(name=collection_name)
            sources_to_delete = vector_storage.get_sources_from_ids(
                collection=collection, source_ids=[prev_document_id]
            )
            vector_storage.delete_sources(
                collection_name=collection_name,
                sources=sources_to_delete,
            )
        # Write uploaded file to disk temporarily
        # @TODO Is there a way to pass this in memory so we dont have to write the file to disk.
        input_file = await copy_file_to_disk(
            app=self.app,
            url_path=url_path,
            file_path=local_file_path,
            text_input=text_input,
            file=file,
            id=source_id,
        )
        # Metadata
        checksum: str = input_file.get("checksum") or ""
        file_name = input_file.get("file_name")
        source_file_name = input_file.get("source_file_name")
        source_file_path = input_file.get("source_file_path")
        extension = os.path.splitext(file_name)[1]  # Remove the dot from the extension
        file_type = extension[1:] or ""
        file_path: str = input_file.get("path_to_file")
        created_at = datetime.now(timezone.utc).strftime("%B %d %Y - %H:%M:%S") or ""
        file_size = 0
        is_file = os.path.isfile(file_path)
        if is_file:
            file_size = os.path.getsize(file_path)
        metadata = {
            "collection_name": collection_name,
            "document_name": document_name,
            "document_id": source_id,
            "embedding_model": self.embed_model_name,
            "description": description,  # transcription of complete source text or summary
            "checksum": checksum,  # the hash of the parsed file
            "tags": tags,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_strategy": chunk_strategy,
            "parsing_method": parsing_method,
            "source_file_name": source_file_name,  # name of the original source file that was parsed
            "source_file_path": source_file_path,
            "file_path": file_path,
            "file_type": file_type,  # type of the source (ingested) file
            "file_size": file_size,  # bytes
            "created_at": created_at,
        }
        # Read in files and create index nodes
        nodes = await self.create_index_nodes(metadata=metadata)
        # Create embeddings
        # @TODO Note that you must NOT perform CPU intensive computations in the background_tasks of the app,
        # because it runs in the same async event loop that serves the requests and it will stall your app.
        # Instead submit them to a thread pool or a process pool.
        background_tasks.add_task(
            self._create_new_embedding,
            vector_storage=vector_storage,
            nodes=nodes,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
            collection_name=collection_name,
        )
        return file_path

    # Create nodes from a single source Document
    async def create_index_nodes(
        self,
        metadata: dict,
    ) -> List[Document]:
        print(f"{common.PRNT_EMBED} Creating nodes...", flush=True)
        # File attributes
        file_path: str = metadata.get("file_path")
        document_id: str = metadata.get("document_id")
        parsing_method: str = metadata.get("parsing_method")
        # Read in source files and build documents
        source_paths = [file_path]
        file_nodes = await documents_from_sources(
            app=self.app,
            sources=source_paths,
            source_id=document_id,
            source_metadata=metadata,
            parsing_method=parsing_method,
        )
        # Optional step, Post-Process source text for optimal embedding/retrieval for LLM
        is_dirty = False  # @TODO Have `documents_from_sources` determine when docs are already processed
        if is_dirty:
            file_nodes = process_documents(nodes=file_nodes)
        return file_nodes
