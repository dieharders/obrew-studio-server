import os
import json
import shutil
from datetime import datetime, timezone
from core import classes, common
from core.download_manager import DownloadManager
from fastapi import (
    APIRouter,
    Request,
    Depends,
    File,
    BackgroundTasks,
    UploadFile,
    HTTPException,
)
from embeddings.vector_storage import VECTOR_STORAGE_PATH, Vector_Storage
from embeddings.embedder import Embedder
from . import file_parsers
from huggingface_hub import hf_hub_download, model_info

router = APIRouter()


def _get_embedding_dim_from_config(model_name: str) -> int | None:
    """Look up embedding dimension from config file based on model name."""
    try:
        config_path = common.dep_path(
            os.path.join("public", "embedding_model_configs.json")
        )
        with open(config_path, "r") as f:
            configs = json.load(f)
        if model_name in configs:
            return configs[model_name].get("dimensions")
    except Exception:
        pass
    return None


@router.get("/addCollection")
def create_memory_collection(
    request: Request,
    form: classes.AddCollectionRequest = Depends(),
) -> classes.AddCollectionResponse:
    app = request.app

    try:
        parsed_tags = common.parse_valid_tags(form.tags)
        collection_name = form.collectionName
        if not collection_name:
            raise Exception("You must supply a collection name.")
        if parsed_tags == None:
            raise Exception("Invalid value for 'tags' input.")
        if not common.check_valid_id(collection_name):
            raise Exception(
                "Invalid collection name. No '--', uppercase, spaces or special chars allowed."
            )
        # Create payload. ChromaDB only accepts strings, numbers, bools.
        # Use specified embedding model or default
        embedder = Embedder(
            app=app,
            embed_model=form.embeddingModel,
        )
        # Look up dimension from config, fall back to passed value for custom models
        config_dim = _get_embedding_dim_from_config(embedder.embed_model_name)
        embedding_dim = config_dim or (
            form.embeddingDim if form.embeddingDim and form.embeddingDim > 0 else None
        )
        metadata = {
            "icon": form.icon or "",
            "created_at": datetime.now(timezone.utc).strftime("%B %d %Y - %H:%M:%S"),
            "tags": parsed_tags,
            "description": form.description,
            "sources": json.dumps([]),  # @TODO need to add the sources
            "embedding_model": embedder.embed_model_name,
            "embedding_dim": embedding_dim,
        }
        vector_storage = Vector_Storage(app=app)
        vector_storage.db_client.create_collection(
            name=collection_name,
            metadata=metadata,
            embedding_function=None,
        )
        msg = f'Successfully created new collection "{collection_name}" with embedding model "{embedder.embed_model_name}"'
        print(f"{common.PRNT_API} {msg}")
        return {
            "success": True,
            "message": msg,
        }
    except Exception as e:
        msg = f'Failed to create new collection "{collection_name}": {e}'
        print(f"{common.PRNT_API} {msg}")
        return {
            "success": False,
            "message": msg,
        }


# Create a memory for Ai.
# This is a multi-step process involving several endpoints.
# It will first process the file, then embed its data into vector space and
# finally add it as a document to specified collection.
@router.post("/addDocument")
async def create_memory(
    request: Request,
    form: classes.EmbedDocumentRequest = Depends(),
    file: UploadFile = File(None),  # File(...) means required
    background_tasks: BackgroundTasks = None,  # This prop is auto populated by FastAPI
) -> classes.AddDocumentResponse:
    tmp_input_file_path = ""
    app = request.app

    try:
        collection_name = form.collectionName
        vector_storage = Vector_Storage(app=app)
        collection = vector_storage.get_collection(name=collection_name)
        embed_model = collection.metadata.get("embedding_model")
        embedder = Embedder(
            app=app,
            embed_model=embed_model,
        )
        vector_storage = Vector_Storage(app=app, embed_fn=embedder.embed_text)
        tmp_input_file_path = await embedder.modify_document(
            vector_storage=vector_storage,
            form=form,
            file=file,
            background_tasks=background_tasks,
            is_update=False,
        )
    except (Exception, KeyError) as e:
        # Error
        msg = f"Failed to create a new memory: {e}"
        print(f"{common.PRNT_API} {msg}")
        return {
            "success": False,
            "message": msg,
        }
    else:
        msg = "A new memory has been added to the queue. It will be available for use shortly."
        print(f"{common.PRNT_API} {msg}", flush=True)
        return {
            "success": True,
            "message": msg,
        }
    finally:
        # Delete uploaded tmp file
        if os.path.exists(tmp_input_file_path):
            os.remove(tmp_input_file_path)
            print(f"{common.PRNT_API} Removed temp file.")


# Re-process and re-embed document
@router.post("/updateDocument")
async def update_memory(
    request: Request,
    form: classes.EmbedDocumentRequest = Depends(),
    file: UploadFile = File(None),  # File(...) means required
    background_tasks: BackgroundTasks = None,  # This prop is auto populated by FastAPI
) -> classes.AddDocumentResponse:
    tmp_input_file_path = ""
    app = request.app

    try:
        collection_name = form.collectionName
        vector_storage = Vector_Storage(app=app)
        collection = vector_storage.get_collection(name=collection_name)
        embed_model = collection.metadata.get("embedding_model")
        embedder = Embedder(
            app=app,
            embed_model=embed_model,
        )
        vector_storage = Vector_Storage(app=app, embed_fn=embedder.embed_text)
        tmp_input_file_path = await embedder.modify_document(
            vector_storage=vector_storage,
            form=form,
            file=file,
            background_tasks=background_tasks,
            is_update=True,
        )
    except (Exception, KeyError) as e:
        # Error
        msg = f"Failed to update memory: {e}"
        print(f"{common.PRNT_API} {msg}", flush=True)
        return {
            "success": False,
            "message": msg,
        }
    else:
        msg = "A memory has been added to the update queue. It will be available for use shortly."
        print(f"{common.PRNT_API} {msg}", flush=True)
        return {
            "success": True,
            "message": msg,
        }
    finally:
        # Delete uploaded tmp file
        if os.path.exists(tmp_input_file_path):
            os.remove(tmp_input_file_path)
            print(f"{common.PRNT_API} Removed temp file.")


@router.get("/getAllCollections")
def get_all_collections(
    request: Request,
) -> classes.GetAllCollectionsResponse:
    app = request.app

    try:
        vector_storage = Vector_Storage(app=app)
        collections = vector_storage.get_all_collections()

        return {
            "success": True,
            "message": f"Returned {len(collections)} collection(s)",
            "data": collections,
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": [],
        }


# Return a collection by id and all its documents
@router.post("/getCollection")
def get_collection(
    request: Request,
    props: classes.GetCollectionRequest,
) -> classes.GetCollectionResponse:
    app = request.app

    try:
        name = props.id
        vector_storage = Vector_Storage(app=app)
        collection = vector_storage.get_collection(name=name)

        return {
            "success": True,
            "message": f"Returned collection(s) {name}",
            "data": vector_storage.collection_to_dict(collection),
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
            "data": {},
        }


# Get all chunks for a source document
@router.post("/getChunks")
def get_chunks(request: Request, params: classes.GetDocumentChunksRequest):
    collection_id = params.collectionId
    document_id = params.documentId
    num_chunks = 0
    app = request.app

    try:
        vector_storage = Vector_Storage(app=app)
        chunks = vector_storage.get_source_chunks(
            collection_name=collection_id, source_id=document_id
        )
        if chunks != None:
            num_chunks = len(chunks)

        return {
            "success": True,
            "message": f"Returned {num_chunks} chunks for document.",
            "data": chunks,
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": f"{e}",
            "data": None,
        }


# Open an OS file exporer on host machine
@router.get("/fileExplore")
def explore_source_file(
    params: classes.FileExploreRequest = Depends(),
) -> classes.FileExploreResponse:
    filePath = params.filePath

    if not os.path.exists(filePath):
        return {
            "success": False,
            "message": "No file path exists",
        }

    # Open a new os window
    common.file_explore(filePath)

    return {
        "success": True,
        "message": "Opened file explorer",
    }


# Delete one or more source documents by id
@router.post("/deleteDocuments")
def delete_document_sources(
    request: Request,
    params: classes.DeleteDocumentsRequest,
) -> classes.DeleteDocumentsResponse:
    app = request.app

    try:
        collection_name = params.collection_id
        source_ids = params.document_ids
        num_documents = len(source_ids)
        # Find source data
        vector_storage = Vector_Storage(app=app)
        collection = vector_storage.get_collection(name=collection_name)
        sources_to_delete = vector_storage.get_sources_from_ids(
            collection=collection, source_ids=source_ids
        )
        # Remove specified source(s)
        vector_storage.delete_sources(
            collection_name=collection_name,
            sources=sources_to_delete,
        )

        return {
            "success": True,
            "message": f"Removed {num_documents} source(s): {source_ids}",
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


# Delete a collection by id
@router.get("/deleteCollection")
def delete_collection(
    request: Request,
    params: classes.DeleteCollectionRequest = Depends(),
) -> classes.DeleteCollectionResponse:
    app = request.app

    try:
        collection_id = params.collection_id
        vector_storage = Vector_Storage(app=app)
        db = vector_storage.db_client
        collection = db.get_collection(collection_id)
        # Remove all the sources in this collection
        sources = vector_storage.get_collection_sources(collection)
        # Remove all associated source files
        vector_storage.delete_sources(
            collection_name=collection_id,
            sources=sources,
        )
        # Remove the collection
        db.delete_collection(name=collection_id)
        # Remove persisted vector index from disk
        common.delete_vector_store(
            target_file_path=collection_id, folder_path=VECTOR_STORAGE_PATH
        )
        return {
            "success": True,
            "message": f"Removed collection [{collection_id}]",
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


# Completely wipe database
@router.get("/wipe")
def wipe_all_memories(
    request: Request,
) -> classes.WipeMemoriesResponse:
    app = request.app

    try:
        # Delete all parsed files in /memories
        file_parsers.delete_all_files()
        # Release the db connection before deleting storage files
        # ChromaDB PersistentClient lacks a close() method (GitHub issue #5868),
        # so we clear the reference and force garbage collection
        if app.state.db_client is not None:
            import gc

            app.state.db_client = None
            gc.collect()
        # Remove all vector storage collections and folders (including chroma.sqlite3)
        Vector_Storage.delete_all_vector_storage()
        # Acknowledge success
        return {
            "success": True,
            "message": "Successfully wiped all memories from Ai",
        }
    except Exception as e:
        print(f"{common.PRNT_API} Error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


# Download an embedding model from huggingface hub (async with progress for GGUF)
@router.post("/downloadEmbedModel")
def download_embedding_model(
    request: Request, payload: classes.DownloadEmbeddingModelRequest
):
    """
    Start an async download of an embedding model.
    For GGUF models: Returns task_id for SSE progress tracking.
    For transformer models: Downloads synchronously (multiple files).
    """
    try:
        app: classes.FastAPIApp = request.app
        download_manager: DownloadManager = app.state.download_manager

        repo_id = payload.repo_id
        filename = payload.filename
        cache_dir = common.app_path(common.EMBEDDING_MODELS_CACHE_DIR)

        # Extract model name from repo_id
        model_name = repo_id.split("/")[-1]

        # Check if this is a GGUF model (single file download with progress)
        is_gguf = filename.lower().endswith(".gguf")

        print(f"{common.PRNT_API} Downloading embedding model {repo_id}...", flush=True)

        if is_gguf:
            # GGUF models use async download with progress tracking
            # NOTE: We save the model metadata only on completion (in on_complete callback)
            # to avoid showing cancelled downloads as "Ready"

            def on_complete(task_id: str, file_path: str):
                """Called when GGUF download completes."""
                try:
                    model_path = os.path.join(
                        cache_dir, f"models--{repo_id.replace('/', '--')}"
                    )
                    total_size = 0
                    if os.path.exists(model_path):
                        for dirpath, _, filenames in os.walk(model_path):
                            for f in filenames:
                                fp = os.path.join(dirpath, f)
                                if os.path.exists(fp):
                                    total_size += os.path.getsize(fp)

                    common.save_embedding_model(
                        {
                            "repoId": repo_id,
                            "modelName": model_name,
                            "savePath": model_path,
                            "size": total_size,
                        }
                    )
                    print(
                        f"{common.PRNT_API} Embedding model saved: {model_path}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"{common.PRNT_API} Error in on_complete: {e}", flush=True)

            def on_error(task_id: str, error: Exception):
                print(
                    f"{common.PRNT_API} Embedding download failed: {error}", flush=True
                )

            task_id = download_manager.start_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir,
                on_complete=on_complete,
                on_error=on_error,
            )

            return {
                "success": True,
                "message": f"Download started for {repo_id}/{filename}",
                "data": {"taskId": task_id},
            }

        else:
            # Transformer models: synchronous multi-file download (no SSE progress)
            # NOTE: Model metadata is only saved after successful completion (below)
            # to avoid showing cancelled/failed downloads as "installed"

            files_to_download = [
                "config.json",
                "tokenizer_config.json",
                "tokenizer.json",
                "special_tokens_map.json",
                "vocab.txt",
                "model.safetensors",
            ]

            downloaded_files = []
            for file in files_to_download:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=file,
                        cache_dir=cache_dir,
                    )
                    downloaded_files.append(file)
                    print(f"{common.PRNT_API} Downloaded {file}", flush=True)
                except Exception:
                    if file == "model.safetensors":
                        try:
                            hf_hub_download(
                                repo_id=repo_id,
                                filename="pytorch_model.bin",
                                cache_dir=cache_dir,
                            )
                            downloaded_files.append("pytorch_model.bin")
                            print(
                                f"{common.PRNT_API} Downloaded pytorch_model.bin",
                                flush=True,
                            )
                        except:
                            pass
                    continue

            if not downloaded_files:
                raise Exception("No model files were successfully downloaded")

            common.scan_cached_repo(cache_dir=cache_dir, repo_id=repo_id)
            model_path = os.path.join(
                cache_dir, f"models--{repo_id.replace('/', '--')}"
            )

            total_size = 0
            if os.path.exists(model_path):
                for dirpath, _, filenames in os.walk(model_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if os.path.exists(fp):
                            total_size += os.path.getsize(fp)

            common.save_embedding_model(
                {
                    "repoId": repo_id,
                    "modelName": model_name,
                    "savePath": model_path,
                    "size": total_size,
                }
            )

            print(f"{common.PRNT_API} Successfully downloaded {repo_id}", flush=True)
            size_mb = total_size / (1024 * 1024)
            return {
                "success": True,
                "message": f"Saved embedding model to {model_path}. Size: {size_mb:.2f} MB",
                "data": None,
            }

    except (KeyError, Exception, EnvironmentError, OSError, ValueError) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


# Return a list of all currently installed embedding models
@router.get("/installedEmbedModels")
def get_installed_embedding_models() -> classes.InstalledEmbeddingModelsResponse:
    try:
        data = []
        # Get installed models file
        metadatas = common.get_settings_file(
            common.APP_SETTINGS_PATH, common.EMBEDDING_METADATAS_FILEPATH
        )
        if not metadatas:
            metadatas = common.DEFAULT_EMBEDDING_SETTINGS_DICT
        if common.INSTALLED_EMBEDDING_MODELS in metadatas:
            data = metadatas[common.INSTALLED_EMBEDDING_MODELS]
            return {
                "success": True,
                "message": f"Returned {len(data)} installed embedding model(s).",
                "data": data,
            }
        else:
            raise Exception(
                f"No attribute {common.INSTALLED_EMBEDDING_MODELS} exists in settings file."
            )
    except Exception as err:
        return {
            "success": False,
            "message": f"Failed to find any installed embedding models. {err}",
            "data": [],
        }


# Returns the curated list of embedding models available for installation
@router.get("/availableEmbedModels")
def get_available_embedding_models() -> classes.EmbeddingModelConfigsResponse:
    try:
        # Get data from file
        config_path = common.dep_path(
            os.path.join("public", "embedding_model_configs.json")
        )
        with open(config_path, "r") as file:
            embedding_models = json.load(file)

        return {
            "success": True,
            "message": f"Returned {len(embedding_models)} available embedding model(s).",
            "data": embedding_models,
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Something went wrong. Reason: {err}",
            "data": {},
        }


# Remove embedding model and installation record
@router.post("/deleteEmbedModel")
def delete_embedding_model(payload: classes.DeleteEmbeddingModelRequest):
    repo_id = payload.repoId

    try:
        cache_dir = common.app_path(common.EMBEDDING_MODELS_CACHE_DIR)

        # Get the model path
        model_path = os.path.join(cache_dir, f"models--{repo_id.replace('/', '--')}")

        if not os.path.exists(model_path):
            raise Exception(f"Model {repo_id} not found in cache")

        # Calculate size before deletion
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)

        # Delete the model directory
        shutil.rmtree(model_path)

        # Delete install record from json file
        common.delete_embedding_model_revisions(repo_id=repo_id)

        # Format size
        size_mb = total_size / (1024 * 1024)
        freed_size_str = f"{size_mb:.2f} MB"

        print(f"{common.PRNT_API} Freed {freed_size_str} space.", flush=True)

        return {
            "success": True,
            "message": f"Deleted embedding model {repo_id}. Freed {freed_size_str} of space.",
            "data": None,
        }
    except (KeyError, Exception) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


# Get embedding model info from HuggingFace
@router.get("/getEmbedModelInfo")
def get_embedding_model_info(params: classes.GetModelInfoRequest = Depends()):
    try:
        repo_id = params.repoId

        # Fetch model info from HuggingFace
        info = model_info(repo_id)

        return {
            "success": True,
            "message": f"Retrieved info for {repo_id}",
            "data": {
                "repoId": repo_id,
                "modelId": info.id,
                "author": info.author,
                "lastModified": str(info.lastModified) if info.lastModified else None,
                "private": info.private,
                "downloads": info.downloads,
                "likes": info.likes,
                "tags": info.tags,
            },
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch model info. Reason: {err}"
        )
