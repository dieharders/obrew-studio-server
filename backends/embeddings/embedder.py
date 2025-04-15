import os
from typing import List, TYPE_CHECKING
from fastapi import BackgroundTasks, UploadFile
from core import common
from core.classes import EmbedDocumentRequest, FastAPIApp
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .file_parsers import copy_file_to_disk, create_parsed_id, process_documents
from .text_splitters import markdown_heading_split, markdown_document_split
from .chunking import chunks_from_documents, create_source_record
from .file_loaders import documents_from_sources

if TYPE_CHECKING:
    from embeddings.vector_storage import Vector_Storage

# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core import StorageContext


embedding_model_names = dict(
    bge_small="BAAI/bge-small-en-v1.5",
    bge_large="BAAI/bge-large-en",
    gte_small="thenlper/gte-small",
    gte_large="thenlper/gte-large",
    gte_base="thenlper/gte-base",  # what we have historically been using
    all_mini="sentence-transformers/all-MiniLM-L6-v2",
    multilingual_large="intfloat/multilingual-e5-large-instruct",  # currently top of leaderboard
)

EMBEDDING_MODEL_CACHE_PATH = common.app_path("embed_models")
CHUNKING_STRATEGIES = {
    "MARKDOWN_HEADING_SPLIT": markdown_heading_split,
    "MARKDOWN_DOCUMENT_SPLIT": markdown_document_split,
}


# How to use local embeddings: https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
# All models: https://huggingface.co/models?library=sentence-transformers
# Rankings: https://huggingface.co/spaces/mteb/leaderboard
class Embedder:
    """Handle vector embeddings."""

    def __init__(self, app: FastAPIApp, model: str = None, cache_path: str = None):
        self.app = app
        self.cache = cache_path or EMBEDDING_MODEL_CACHE_PATH
        self.embed_model_name = model or embedding_model_names["multilingual_large"]
        self.embed_model = HuggingFaceEmbedding(
            self.embed_model_name, cache_folder=self.cache
        )

    # Create embeddings for one file (input) at a time
    # We create one source per upload
    def _create_new_embedding(
        self,
        vector_storage: "Vector_Storage",
        nodes: List[Document],
        form: dict,
    ):
        try:
            print(f"{common.PRNT_EMBED} Creating embeddings...", flush=True)
            # File attributes
            chunk_size: int = form["chunk_size"] or 300
            chunk_overlap: int = form["chunk_overlap"] or 0
            chunk_strategy: str = (
                form["chunk_strategy"] or list(CHUNKING_STRATEGIES.keys())[0]
            )
            text_splitter = CHUNKING_STRATEGIES[chunk_strategy]
            collection_name: str = form["collection_name"]
            # Loop through each file
            for node in nodes:
                # Create source document records for Collection metadata
                source_record = create_source_record(document=node)
                # Split document texts
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
                    vector_index = vector_storage.add_chunks_to_collection(
                        collection=collection,
                        nodes=chunk_nodes,
                        callback_manager=self.create_index_callback_manager(),
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

    # Create document embeddings from chunks, not currently used.
    def _embed_pipeline(self, parser, vector_store, documents: List[Document]):
        pipeline = IngestionPipeline(
            transformations=[parser],
            vector_store=vector_store,
            docstore=SimpleDocumentStore(),  # enabled document management
        )
        # Ingest directly into a vector db
        pipeline.run(documents=documents)

    def embed_text(self, text: str) -> List[float]:
        """Create vector embeddings from query text."""
        embedding_model = HuggingFaceEmbedding(
            self.embed_model_name, cache_folder=self.cache
        )
        # Both work
        return embedding_model.get_text_embedding(text)
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
        chunk_size = form.chunkSize
        chunk_overlap = form.chunkOverlap
        chunk_strategy = form.chunkStrategy
        parsing_method = form.parsingMethod
        new_document_id = create_parsed_id(collection_name=collection_name)
        if is_update:
            source_id = prev_document_id
        else:
            source_id = new_document_id
        form_data = {
            "collection_name": collection_name,
            "document_name": document_name,
            "document_id": source_id,
            "description": description,
            "tags": tags,
            "embedder": self.embed_model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_strategy": chunk_strategy,
            "parsing_method": parsing_method,
        }
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
            vector_index = self.load_embedding(
                collection_name=collection_name, vector_storage=vector_storage
            )
            vector_storage.delete_sources(
                collection_name=collection_name,
                sources=sources_to_delete,
                vector_storage=vector_storage,
                vector_index=vector_index,
            )
        # Write uploaded file to disk temporarily
        input_file = await copy_file_to_disk(
            app=self.app,
            url_path=url_path,
            file_path=local_file_path,
            text_input=text_input,
            file=file,
            id=source_id,
        )
        path_to_parsed_file = input_file.get("path_to_file")
        # Read in files and create index nodes
        nodes = await self.create_index_nodes(
            input_file=input_file,
            form=form_data,
        )
        # Create embeddings
        # @TODO Note that you must NOT perform CPU intensive computations in the background_tasks of the app,
        # because it runs in the same async event loop that serves the requests and it will stall your app.
        # Instead submit them to a thread pool or a process pool.
        background_tasks.add_task(
            self._create_new_embedding,
            nodes=nodes,
            form=form_data,
            vector_storage=vector_storage,
        )
        return path_to_parsed_file

    # Load collection of document embeddings from disk
    # @TODO Cleanup args, some (context_window, num_output, chunk_size, prompts) should be updated before a query is called
    def load_embedding(
        self, collection_name: str, vector_storage: "Vector_Storage"
    ) -> VectorStoreIndex:
        vector_index = None
        try:
            # Get collection
            collection = vector_storage.get_collection(name=collection_name)
            if collection:
                vector_index = vector_storage.add_chunks_to_collection(
                    collection=collection,
                    nodes=[],  # empty since we're just loading
                    callback_manager=self.create_index_callback_manager(),
                )
        except Exception as e:
            print(f"{common.PRNT_EMBED} Failed to load vector index: {e}", flush=True)
        return vector_index

    # Create nodes from a single source Document
    async def create_index_nodes(
        self,
        input_file: dict,
        form: dict,
    ) -> List[Document]:
        print(f"{common.PRNT_EMBED} Creating nodes...", flush=True)
        # File attributes
        checksum: str = input_file.get("checksum")
        file_name = input_file.get("file_name")
        source_file_path: str = input_file.get("path_to_file")
        document_id: str = form["document_id"]
        document_name: str = form["document_name"]
        description: str = form["description"]
        tags: str = form["tags"]
        parsing_method: str = form["parsing_method"]
        is_file = os.path.isfile(source_file_path)
        file_size = 0
        if is_file:
            file_size = os.path.getsize(source_file_path)
        # Read in source files and build documents
        source_paths = [source_file_path]
        source_metadata = dict(
            name=document_name,
            description=description,
            checksum=checksum,
            fileName=file_name,
            filePath=source_file_path,
            fileSize=file_size,
            tags=tags,
        )
        file_nodes = await documents_from_sources(
            app=self.app,
            sources=source_paths,
            source_id=document_id,
            source_metadata=source_metadata,
            parsing_method=parsing_method,
        )
        # Optional step, Post-Process source text for optimal embedding/retrieval for LLM
        is_dirty = False  # @TODO Have `documents_from_sources` determine when docs are already processed
        if is_dirty:
            file_nodes = process_documents(nodes=file_nodes)
        return file_nodes

    # For debugging llama-index events
    def create_index_callback_manager(self) -> CallbackManager:
        # Debugging - https://docs.llamaindex.ai/en/v0.10.19/examples/callbacks/LlamaDebugHandler.html
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        return callback_manager
