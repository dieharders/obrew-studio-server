import os
import json
import uuid
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from .image_embedder import ImageEmbedder
from core import classes, common
from core.common import get_model_install_config
from core.download_manager import DownloadManager
from inference.llama_cpp import LLAMA_CPP
from inference.classes import (
    CHAT_MODES,
    AgentOutput,
    LoadInferenceResponse,
    SSEResponse,
    VisionInferenceRequest,
    LoadVisionInferenceRequest,
    VisionEmbedRequest,
    VisionEmbedLoadRequest,
    DownloadVisionEmbedModelRequest,
    DeleteVisionEmbedModelRequest,
    VisionEmbedQueryRequest,
)
from inference.helpers import (
    decode_base64_image,
    cleanup_temp_images,
    preprocess_image,
)
from .image_embedder import VISION_EMBEDDING_MODELS_CACHE_DIR
from embeddings.vector_storage import Vector_Storage


router = APIRouter()

# ============================================================================
# Image/Multi-Modal Inference Endpoints
# ============================================================================


# Load vision (inference) model with mmproj
@router.post("/load")
async def load_vision_model(
    request: Request,
    data: LoadVisionInferenceRequest,
) -> LoadInferenceResponse:
    """Load a vision model with its mmproj file. Stores in app.state.llm object."""
    app: classes.FastAPIApp = request.app

    try:
        model_id = data.modelId

        # Look up paths from model_id if not provided
        model_path = data.modelPath or common.get_model_file_path(model_id)
        mmproj_path = data.mmprojPath or common.get_mmproj_path(model_id)

        # Validate paths exist
        if not model_path:
            raise Exception(
                f"Model path not found for {model_id}. Provide modelPath or install the model first."
            )
        if not mmproj_path:
            raise Exception(
                f"mmproj path not found for {model_id}. Provide mmprojPath or download the mmproj first."
            )
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        if not os.path.exists(mmproj_path):
            raise Exception(f"mmproj file not found: {mmproj_path}")

        # Unload existing model if present (unified state)
        if hasattr(app.state, "llm") and app.state.llm:
            print(
                f"{common.PRNT_API} Ejecting current model before loading vision model"
            )
            await app.state.llm.unload()
            app.state.llm = None

        # Get model config
        model_config = get_model_install_config(model_id)
        model_name = model_config.get("model_name")

        # Create unified model instance with mmproj for vision capability
        app.state.llm = LLAMA_CPP(
            model_path=model_path,
            mmproj_path=mmproj_path,
            model_name=model_name,
            model_id=model_id,
            response_mode=CHAT_MODES.INSTRUCT,  # @TODO Should be passed in or set as default val
            model_init_kwargs=data.init,
            generate_kwargs=data.call,
        )

        print(f"{common.PRNT_API} Vision model {model_id} loaded")
        return {
            "message": f"Vision model [{model_id}] loaded.",
            "success": True,
            "data": None,
        }
    except Exception as error:
        print(f"{common.PRNT_API} Failed loading vision model: {error}")
        return {
            "message": f"Unable to load vision model [{data.modelId}]: {error}",
            "success": False,
            "data": None,
        }


# Unload vision (inference) model
@router.post("/unload")
async def unload_vision_model(request: Request):
    """Unload the currently loaded model (unified state)."""
    try:
        app: classes.FastAPIApp = request.app
        if hasattr(app.state, "llm") and app.state.llm:
            await app.state.llm.unload()
        app.state.llm = None
        return {
            "success": True,
            "message": "Vision model was unloaded",
            "data": None,
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error unloading vision model: {err}", flush=True)
        return {
            "success": False,
            "message": f"Error unloading vision model: {err}",
            "data": None,
        }


# Run vision inference
@router.post("/generate")
async def generate_vision(
    request: Request,
    payload: VisionInferenceRequest,
) -> SSEResponse | AgentOutput | classes.GenericEmptyResponse:
    """Run vision inference on image(s) with a text prompt."""
    app: classes.FastAPIApp = request.app
    temp_image_paths = []

    try:
        # Check if model with vision capability is loaded
        if not hasattr(app.state, "llm") or not app.state.llm:
            raise Exception("No model loaded. Call /vision/load first.")

        llm = app.state.llm

        # Check if model has mmproj (vision capability)
        if not llm.mmproj_path:
            raise Exception(
                "Model not loaded with vision capability. "
                "Use /vision/load to load a model with mmproj."
            )

        # Validate images array is not empty
        if not payload.images:
            raise HTTPException(status_code=400, detail="No images provided")

        # Build override args from vision request (don't use generate_kwargs setter
        # since VisionInferenceRequest has different fields than InferenceRequest)
        override_args = {
            "--temp": payload.temperature,
            "--n-predict": payload.max_tokens,
            "--top-k": payload.top_k,
            "--top-p": payload.top_p,
            "--min-p": payload.min_p,
            "--repeat-penalty": payload.repeat_penalty,
        }

        # Process images
        temp_dir = tempfile.gettempdir()
        image_paths = []

        for img in payload.images:
            if payload.image_type == "base64":
                # Decode base64 to temp file
                path = decode_base64_image(img, temp_dir)
                temp_image_paths.append(path)  # Track for cleanup
            else:
                # Use file path directly
                if not os.path.exists(img):
                    raise Exception(f"Image file not found: {img}")
                path = img

            # Preprocess image for llama.cpp compatibility
            # This fixes issues with macOS screenshots and resizes large images
            processed_path = preprocess_image(path, temp_dir)
            if processed_path != path:
                temp_image_paths.append(processed_path)  # Track for cleanup
            image_paths.append(processed_path)

        if not image_paths:
            raise Exception("No valid images provided.")

        # Run vision inference
        response_gen = await llm.vision_completion(
            prompt=payload.prompt,
            image_paths=image_paths,
            request=request,
            system_message=payload.systemMessage,
            stream=payload.stream,
            override_args=override_args,
        )

        # Handle streaming vs non-streaming response
        if payload.stream:
            # Return streaming SSE response
            return response_gen
        else:
            # For non-streaming, consume the generator and return final text
            text = ""
            async for chunk in response_gen:
                if isinstance(chunk, dict):
                    # content_payload returns {"event": "...", "data": {"text": text}}
                    if "data" in chunk and isinstance(chunk["data"], dict):
                        text = chunk["data"].get("text", text)
                elif isinstance(chunk, str):
                    try:
                        parsed = json.loads(chunk)
                        if "data" in parsed and isinstance(parsed["data"], dict):
                            text = parsed["data"].get("text", text)
                    except json.JSONDecodeError:
                        pass

            return {"success": True, "text": text, "data": None}

    except Exception as err:
        err_type = type(err).__name__
        err_msg = str(err) if str(err) else "Unknown error (no message)"
        print(
            f"{common.PRNT_API} Vision generation error [{err_type}]: {err_msg}",
            flush=True,
        )
        return {
            "success": False,
            "message": f"Vision generation failed: {err_msg}",
            "data": None,
        }
    finally:
        # Cleanup temporary images
        if temp_image_paths:
            cleanup_temp_images(temp_image_paths)


# Get currently loaded vision (inference) model info
@router.get("/model")
def get_vision_model(request: Request):
    """Get info about the currently loaded vision model."""
    app: classes.FastAPIApp = request.app

    try:
        # Check if model with vision capability is loaded
        if hasattr(app.state, "llm") and app.state.llm and app.state.llm.mmproj_path:
            llm = app.state.llm
            return {
                "success": True,
                "message": f"{llm.model_id} is loaded with vision capability.",
                "data": {
                    "modelId": llm.model_id,
                    "modelName": llm.model_name,
                    "modelPath": llm.model_path,
                    "mmprojPath": llm.mmproj_path,
                },
            }
        else:
            return {
                "success": False,
                "message": "No model with vision capability is currently loaded.",
                "data": {},
            }
    except Exception as error:
        return {
            "success": False,
            "message": f"Error getting vision model info: {error}",
            "data": {},
        }


# ============================================================================
# Image/Multi-Modal Embedding Endpoints
# ============================================================================


def _get_vision_embedder(app: classes.FastAPIApp) -> ImageEmbedder:
    """Get or create the vision embedder instance."""
    if not hasattr(app.state, "vision_embedder") or app.state.vision_embedder is None:
        app.state.vision_embedder = ImageEmbedder(app)
    return app.state.vision_embedder


@router.post("/embed/load")
async def load_embedding_model(
    request: Request,
    data: VisionEmbedLoadRequest,
):
    """
    Load a multimodal embedding model for image embeddings.
    This starts a separate llama-server process with --embedding flag.
    """
    app: classes.FastAPIApp = request.app

    try:
        embedder = _get_vision_embedder(app)

        success = await embedder.load_model(
            model_path=data.model_path,
            mmproj_path=data.mmproj_path,
            model_name=data.model_name,
            model_id=data.model_id,
            port=data.port,
            n_gpu_layers=data.n_gpu_layers,
            n_threads=data.n_threads,
            n_ctx=data.n_ctx,
        )

        if success:
            return {
                "success": True,
                "message": f"Embedding model loaded: {embedder.model_name}",
                "data": embedder.get_model_info(),
            }
        else:
            return {
                "success": False,
                "message": "Failed to load embedding model",
                "data": None,
            }
    except Exception as err:
        print(f"{common.PRNT_API} Error loading embedding model: {err}", flush=True)
        return {
            "success": False,
            "message": f"Failed to load embedding model: {err}",
            "data": None,
        }


@router.post("/embed/unload")
async def unload_embedding_model(request: Request):
    """Unload the currently loaded image embedding model."""
    app: classes.FastAPIApp = request.app

    try:
        embedder = _get_vision_embedder(app)
        await embedder.unload()

        return {
            "success": True,
            "message": "Embedding model unloaded",
            "data": None,
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error unloading embedding model: {err}", flush=True)
        return {
            "success": False,
            "message": f"Failed to unload embedding model: {err}",
            "data": None,
        }


@router.get("/embed/model")
def get_embedding_model(request: Request):
    """Get info about the currently loaded image embedding model."""
    app: classes.FastAPIApp = request.app

    try:
        embedder = _get_vision_embedder(app)
        info = embedder.get_model_info()

        return {
            "success": True,
            "message": "Embedding model info retrieved",
            "data": info,
        }
    except Exception as err:
        return {
            "success": False,
            "message": f"Error getting embedding model info: {err}",
            "data": None,
        }


@router.post("/embed")
async def embed_image(
    request: Request,
    payload: VisionEmbedRequest,
):
    """
    Create embedding for an image and optionally store in ChromaDB.

    The image can be provided as a file path or base64 encoded string.
    If include_transcription is True, the image will also be transcribed
    using the vision model and the transcription stored as metadata.
    """
    app: classes.FastAPIApp = request.app
    temp_image_path = None
    temp_processed_path = None

    try:
        embedder = _get_vision_embedder(app)

        # Get image path
        temp_dir = tempfile.gettempdir()
        if payload.image_type == "base64":
            if not payload.image_base64:
                raise HTTPException(
                    status_code=400,
                    detail="image_base64 is required when image_type is 'base64'",
                )
            # Decode base64 to temp file
            temp_image_path = decode_base64_image(payload.image_base64, temp_dir)
            image_path = temp_image_path
        else:
            if not payload.image_path:
                raise HTTPException(
                    status_code=400,
                    detail="image_path is required when image_type is 'path'",
                )
            if not os.path.exists(payload.image_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"Image file not found: {payload.image_path}",
                )
            image_path = payload.image_path

        # Preprocess image for llama.cpp compatibility
        # This fixes issues with macOS screenshots and resizes large images
        processed_path = preprocess_image(image_path, temp_dir)
        if processed_path != image_path:
            temp_processed_path = processed_path
        image_path = processed_path

        # Get embedding (auto_unload=False so we can capture model_name before unloading)
        print(
            f"{common.PRNT_API} Creating embedding for image: {image_path}", flush=True
        )
        embeddings = await embedder.embed_images([image_path], auto_unload=False)
        embedding = embeddings[0]
        embedding_dim = len(embedding)
        # Capture model name before unloading
        embedding_model_name = embedder.model_name or "unknown"
        await embedder.unload()

        # Determine collection name
        collection_name = payload.collection_name
        if not collection_name:
            # Auto-create from filename
            collection_name = Path(image_path).stem
            # Sanitize collection name (ChromaDB requirements)
            collection_name = "".join(
                c if c.isalnum() or c in "_-" else "_" for c in collection_name
            )
            if not collection_name or collection_name[0].isdigit():
                collection_name = f"images_{collection_name}"

        # Prepare metadata (embedding_model and embedding_dim are stored at collection level)
        source_file_name = Path(image_path).name
        metadata = {
            "type": "image",
            "source_collection_id": collection_name,  # Collection this document belongs to
            "source_file_name": source_file_name,
            "source_file_path": payload.image_path or "base64_upload",
            "description": payload.description
            or "",  # Transcription/description of the image
            "created_at": datetime.now().isoformat(),
            **(payload.metadata or {}),
        }

        # Add description to metadata if available
        # Note: description is passed to collection.add() as documents parameter

        # Store in ChromaDB - ensure db_client is initialized
        doc_id = str(uuid.uuid4())

        # Initialize db_client if not already done (Vector_Storage handles this)
        if not hasattr(app.state, "db_client") or not app.state.db_client:
            Vector_Storage(app=app)  # This initializes app.state.db_client

        if app.state.db_client:
            try:
                # Get or create collection
                # Store embedding_model and embedding_dim at collection level for validation
                collection = app.state.db_client.get_or_create_collection(
                    name=collection_name,
                    metadata={
                        "type": "image_embeddings",
                        "embedding_model": embedding_model_name,
                        "embedding_dim": embedding_dim,
                        "description": "",  # Summary of images in this collection
                    },
                )

                # Add to collection
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    # Add raw description text for reference
                    documents=[payload.description or ""],
                )

                print(
                    f"{common.PRNT_API} Stored embedding in collection: {collection_name}",
                    flush=True,
                )
            except Exception as store_err:
                print(
                    f"{common.PRNT_API} Failed to store embedding (non-fatal): {store_err}",
                    flush=True,
                )

        return {
            "success": True,
            "message": "Image embedding created successfully",
            "data": {
                "id": doc_id,
                "collection_name": collection_name,
                "embedding_model": embedding_model_name,
                "embedding_dim": embedding_dim,
                "description": payload.description,
                "metadata": metadata,
            },
        }

    except HTTPException:
        raise
    except Exception as err:
        print(f"{common.PRNT_API} Error creating image embedding: {err}", flush=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to create image embedding: {err}"
        )
    finally:
        # Cleanup temp images
        if temp_image_path and os.path.exists(temp_image_path):
            try:
                os.unlink(temp_image_path)
            except Exception:
                pass
        if temp_processed_path and os.path.exists(temp_processed_path):
            try:
                os.unlink(temp_processed_path)
            except Exception:
                pass


@router.post("/embed/download")
def download_vision_embedding_model(
    request: Request, payload: DownloadVisionEmbedModelRequest
):
    """
    Download a vision embedding model (GGUF + mmproj) from HuggingFace.

    Returns task_id(s) immediately. Use /downloads/progress?task_id=<id> for SSE updates.
    Both model and mmproj downloads start in parallel if mmproj is specified.
    """
    try:
        app: classes.FastAPIApp = request.app
        download_manager: DownloadManager = app.state.download_manager

        repo_id = payload.repo_id
        filename = payload.filename
        mmproj_filename = payload.mmproj_filename
        cache_dir = VISION_EMBEDDING_MODELS_CACHE_DIR

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        print(
            f"{common.PRNT_API} Starting vision embedding model download: {repo_id}",
            flush=True,
        )

        # Thread-safe tracking for parallel downloads
        # Each callback saves independently; metadata is saved when both complete
        download_lock = threading.Lock()
        download_results = {"model_path": None, "mmproj_path": None}

        def _save_metadata_if_complete():
            """Save metadata when both downloads complete. Must be called with lock held."""
            if download_results["model_path"] and download_results["mmproj_path"]:
                try:
                    total_size = 0
                    if os.path.exists(download_results["model_path"]):
                        total_size += os.path.getsize(download_results["model_path"])
                    if os.path.exists(download_results["mmproj_path"]):
                        total_size += os.path.getsize(download_results["mmproj_path"])

                    common.save_vision_embedding_model(
                        repo_id=repo_id,
                        model_path=download_results["model_path"],
                        mmproj_path=download_results["mmproj_path"],
                        size=total_size,
                    )
                    print(
                        f"{common.PRNT_API} Vision embedding model saved: {repo_id}",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"{common.PRNT_API} Error saving vision model metadata: {e}",
                        flush=True,
                    )

        def on_model_complete(task_id: str, file_path: str):
            """Called when main model download completes."""
            try:
                [model_cache_info, repo_revisions] = common.scan_cached_repo(
                    cache_dir=cache_dir, repo_id=repo_id
                )
                model_path = common.get_cached_blob_path(
                    repo_revisions=repo_revisions, filename=filename
                )
                if not isinstance(model_path, str):
                    model_path = file_path

                print(
                    f"{common.PRNT_API} Vision model downloaded: {model_path}",
                    flush=True,
                )
                with download_lock:
                    download_results["model_path"] = model_path
                    _save_metadata_if_complete()
            except Exception as e:
                print(f"{common.PRNT_API} Error in on_model_complete: {e}", flush=True)

        def on_mmproj_complete(task_id: str, file_path: str):
            """Called when mmproj download completes."""
            try:
                [mmproj_cache_info, repo_revisions] = common.scan_cached_repo(
                    cache_dir=cache_dir, repo_id=repo_id
                )
                mmproj_path = common.get_cached_blob_path(
                    repo_revisions=repo_revisions, filename=mmproj_filename
                )
                if not isinstance(mmproj_path, str):
                    mmproj_path = file_path

                print(
                    f"{common.PRNT_API} Vision mmproj downloaded: {mmproj_path}",
                    flush=True,
                )
                with download_lock:
                    download_results["mmproj_path"] = mmproj_path
                    _save_metadata_if_complete()
            except Exception as e:
                print(f"{common.PRNT_API} Error in on_mmproj_complete: {e}", flush=True)

        def on_model_error(task_id: str, error: Exception):
            print(
                f"{common.PRNT_API} Vision model download failed: {error}", flush=True
            )

        def on_mmproj_error(task_id: str, error: Exception):
            print(
                f"{common.PRNT_API} Vision mmproj download failed: {error}", flush=True
            )

        # Start main model download
        task_id = download_manager.start_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            on_complete=on_model_complete,
            on_error=on_model_error,
        )

        # Start mmproj download in parallel
        mmproj_task_id = None
        if mmproj_filename:
            mmproj_task_id = download_manager.start_download(
                repo_id=repo_id,
                filename=mmproj_filename,
                cache_dir=cache_dir,
                on_complete=on_mmproj_complete,
                on_error=on_mmproj_error,
            )
            # Link the tasks so client can track both via main task's progress
            download_manager.set_secondary_task(task_id, mmproj_task_id)

        response_data = {"taskId": task_id}
        if mmproj_task_id:
            response_data["mmprojTaskId"] = mmproj_task_id

        message = f"Download started for {repo_id}"
        if mmproj_task_id:
            message += " (model and mmproj downloading in parallel)"

        return {
            "success": True,
            "message": message,
            "data": response_data,
        }
    except Exception as err:
        print(
            f"{common.PRNT_API} Error downloading vision embedding model: {err}",
            flush=True,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to download vision embedding model: {err}",
        )


@router.get("/embed/installed")
def get_installed_vision_embedding_models():
    """
    Get list of all installed vision embedding models.

    Returns the metadata for all vision embedding models that have been
    downloaded, including repo ID, model path, mmproj path, and size.
    """
    try:
        installed = common.get_installed_vision_embedding_models()
        return {
            "success": True,
            "message": f"Found {len(installed)} installed vision embedding models",
            "data": installed,
        }
    except Exception as err:
        print(
            f"{common.PRNT_API} Error getting installed vision embedding models: {err}",
            flush=True,
        )
        return {
            "success": False,
            "message": f"Failed to get installed models: {err}",
            "data": [],
        }


@router.post("/embed/delete")
def delete_vision_embedding_model(payload: DeleteVisionEmbedModelRequest):
    """
    Delete a vision embedding model (GGUF + mmproj) from local storage.

    This deletes both the main model file and the mmproj file, as well as
    the metadata record.
    """
    try:
        repo_id = payload.repo_id
        cache_dir = VISION_EMBEDDING_MODELS_CACHE_DIR

        print(
            f"{common.PRNT_API} Deleting vision embedding model: {repo_id}",
            flush=True,
        )

        # Get the model paths from metadata before deleting
        model_path, mmproj_path = common.get_vision_embedding_model_path(repo_id)

        # Delete files from HF cache using the cache deletion mechanism
        try:
            [model_cache_info, repo_revisions] = common.scan_cached_repo(
                cache_dir=cache_dir, repo_id=repo_id
            )
            repo_commit_hashes = [r.commit_hash for r in repo_revisions]
            if repo_commit_hashes:
                delete_strategy = model_cache_info.delete_revisions(*repo_commit_hashes)
                delete_strategy.execute()
                freed_size = delete_strategy.expected_freed_size_str
                print(f"{common.PRNT_API} Freed {freed_size} space.", flush=True)
        except Exception as cache_err:
            print(
                f"{common.PRNT_API} Warning: Could not delete from HF cache: {cache_err}",
                flush=True,
            )
            # Try direct file deletion as fallback
            if model_path and os.path.exists(model_path):
                os.remove(model_path)
                print(f"{common.PRNT_API} Deleted model file: {model_path}", flush=True)
            if mmproj_path and os.path.exists(mmproj_path):
                os.remove(mmproj_path)
                print(
                    f"{common.PRNT_API} Deleted mmproj file: {mmproj_path}", flush=True
                )

        # Delete metadata record
        common.delete_vision_embedding_model(repo_id)

        return {
            "success": True,
            "message": f"Deleted vision embedding model: {repo_id}",
            "data": None,
        }
    except Exception as err:
        print(
            f"{common.PRNT_API} Error deleting vision embedding model: {err}",
            flush=True,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Failed to delete vision embedding model: {err}",
        )


@router.post("/embed/query")
async def query_image_collection(
    request: Request,
    payload: VisionEmbedQueryRequest,
):
    """
    Query an image collection using text similarity search.

    This endpoint:
    1. Loads the vision embedding model (auto-loads from cache if needed)
    2. Creates a text embedding for the query using the SAME model
       that created the image embeddings
    3. Performs ChromaDB similarity search on the specified collection
    4. Returns matching images with similarity scores and metadata

    IMPORTANT: The query embedding must be created with the same model
    that was used to create the image embeddings, otherwise the embeddings
    will not be in the same vector space and results will be meaningless.
    """
    app: classes.FastAPIApp = request.app

    try:
        # Validate collection exists and is an image collection
        if not hasattr(app.state, "db_client") or not app.state.db_client:
            Vector_Storage(app=app)  # Initialize db_client

        if not app.state.db_client:
            raise HTTPException(status_code=500, detail="Vector database not available")

        # Get collection
        try:
            collection = app.state.db_client.get_collection(
                name=payload.collection_name
            )
        except Exception as e:
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{payload.collection_name}' not found: {e}",
            )

        # Verify this is an image collection
        collection_type = collection.metadata.get("type", "")
        if collection_type != "image_embeddings":
            print(
                f"{common.PRNT_API} Warning: Collection '{payload.collection_name}' "
                f"has type '{collection_type}', expected 'image_embeddings'",
                flush=True,
            )

        # Check collection is not empty
        if collection.count() == 0:
            return {
                "success": True,
                "message": f"Collection '{payload.collection_name}' is empty",
                "data": {
                    "query": payload.query,
                    "collection_name": payload.collection_name,
                    "results": [],
                    "total_in_collection": 0,
                },
            }

        # Get the vision embedder and create query embedding
        embedder = _get_vision_embedder(app)

        print(
            f"{common.PRNT_API} Creating query embedding for: {payload.query[:100]}...",
            flush=True,
        )

        # Generate query embedding using vision model (auto-loads if needed)
        query_embedding = await embedder.embed_query_text(
            text=payload.query,
            auto_unload=False,  # Keep loaded for potential subsequent queries
        )

        # Capture model info for response
        query_model_name = embedder.model_name or "unknown"

        # Capture query embedding dimension
        query_embedding_dim = len(query_embedding)

        # Validate model consistency: the query model should match the collection's embedding model
        collection_embedding_model = collection.metadata.get("embedding_model")
        if (
            collection_embedding_model
            and collection_embedding_model != query_model_name
        ):
            print(
                f"{common.PRNT_API} Warning: Model mismatch! Collection was embedded with "
                f"'{collection_embedding_model}' but querying with '{query_model_name}'. "
                f"Results may be inaccurate.",
                flush=True,
            )

        # Validate embedding dimension consistency
        collection_embedding_dim = collection.metadata.get("embedding_dim")
        if collection_embedding_dim and collection_embedding_dim != query_embedding_dim:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Embedding dimension mismatch: collection '{payload.collection_name}' "
                    f"has dimension {collection_embedding_dim}, but query embedding has "
                    f"dimension {query_embedding_dim}. Use the same model that created "
                    f"the collection embeddings."
                ),
            )

        print(
            f"{common.PRNT_API} Query embedding dimension: {query_embedding_dim}",
            flush=True,
        )

        # Perform ChromaDB similarity search
        include_fields = ["metadatas", "documents", "distances"]
        if payload.include_embeddings:
            include_fields.append("embeddings")

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=payload.top_k,
            include=include_fields,
        )

        # Format results
        formatted_results = []
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        documents = results.get("documents", [[]])[0]
        embeddings = (
            results.get("embeddings", [[]])[0] if payload.include_embeddings else []
        )

        for i, doc_id in enumerate(ids):
            # Calculate similarity score from distance
            # ChromaDB uses L2 (squared Euclidean) distance by default, which is unbounded [0, inf)
            # Using 1/(1+distance) gives a bounded similarity score in [0, 1]
            # where 0 distance = 1.0 (identical) and higher distance approaches 0
            distance = distances[i] if i < len(distances) else None
            similarity_score = 1.0 / (1.0 + distance) if distance is not None else None

            result_item = {
                "id": doc_id,
                "distance": distance,
                "similarity_score": similarity_score,
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "document": (
                    documents[i] if i < len(documents) else None
                ),  # Contains description if available
            }

            if payload.include_embeddings and i < len(embeddings):
                result_item["embedding"] = embeddings[i]

            formatted_results.append(result_item)

        print(
            f"{common.PRNT_API} Found {len(formatted_results)} results in collection '{payload.collection_name}'",
            flush=True,
        )

        # Build response with optional model mismatch warning
        model_mismatch = (
            collection_embedding_model
            and collection_embedding_model != query_model_name
        )
        message = f"Found {len(formatted_results)} matching images"
        if model_mismatch:
            message += (
                f" (Warning: model mismatch - collection uses '{collection_embedding_model}', "
                f"query uses '{query_model_name}')"
            )

        return {
            "success": True,
            "message": message,
            "data": {
                "query": payload.query,
                "collection_name": payload.collection_name,
                "query_model": query_model_name,
                "collection_model": collection_embedding_model,
                "model_mismatch": model_mismatch,
                "embedding_dim": query_embedding_dim,
                "results": formatted_results,
                "total_in_collection": collection.count(),
            },
        }

    except HTTPException:
        raise
    except Exception as err:
        print(f"{common.PRNT_API} Error querying image collection: {err}", flush=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to query image collection: {err}"
        )
