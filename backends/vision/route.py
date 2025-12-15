import os
import json
import uuid
import tempfile
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, Request, HTTPException
from .llama_cpp_vision import LLAMA_CPP_VISION
from .image_embedder import ImageEmbedder
from core import classes, common
from huggingface_hub import hf_hub_download
from inference.classes import (
    AgentOutput,
    LoadInferenceResponse,
    LoadTextInferenceInit,
    LoadTextInferenceCall,
    SSEResponse,
    VisionInferenceRequest,
    LoadVisionInferenceRequest,
    DownloadMmprojRequest,
    VisionEmbedRequest,
    VisionEmbedLoadRequest,
    DownloadVisionEmbedModelRequest,
    DeleteVisionEmbedModelRequest,
)
from inference.helpers import decode_base64_image, cleanup_temp_images, preprocess_image
from .image_embedder import VISION_EMBEDDING_MODELS_CACHE_DIR


def get_model_install_config(model_id: str = None) -> dict:
    try:
        # Get the config for the model
        config_path = common.dep_path(os.path.join("public", "text_model_configs.json"))
        with open(config_path, "r") as file:
            text_models = json.load(file)
            if not model_id:
                return dict(models=text_models)
            config = text_models[model_id]
            message_format = config["messageFormat"]
            model_name = config["name"]
            tags = config.get("tags")
            repoId = config.get("repoId", "")
            description = config.get("description", "")
            return dict(
                message_format=message_format,
                description=description,
                id=repoId,
                model_name=model_name,
                models=text_models,
                tags=tags,
            )
    except Exception as err:
        raise Exception(f"Error finding models list: {err}")


router = APIRouter()

# ============================================================================
# Image/Multi-Modal Inference Endpoints
# ============================================================================


# Download mmproj file for vision model
@router.post("/download/mmproj")
def download_mmproj(payload: DownloadMmprojRequest):
    """Download mmproj (multimodal projector) file for vision model."""
    try:
        repo_id = payload.repo_id
        filename = payload.filename
        model_repo_id = payload.model_repo_id
        cache_dir = common.app_path(common.TEXT_MODELS_CACHE_DIR)

        # Download mmproj file
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=False,
        )

        # Get actual file path from cache
        [model_cache_info, repo_revisions] = common.scan_cached_repo(
            cache_dir=cache_dir, repo_id=repo_id
        )
        file_path = common.get_cached_blob_path(
            repo_revisions=repo_revisions, filename=filename
        )

        if not isinstance(file_path, str):
            raise Exception("mmproj path is not a string.")

        # Save mmproj path to model metadata
        common.save_mmproj_path(model_repo_id=model_repo_id, mmproj_path=file_path)

        return {
            "success": True,
            "message": f"Downloaded mmproj to {file_path}",
            "data": {"mmprojPath": file_path},
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error downloading mmproj: {err}", flush=True)
        raise HTTPException(status_code=400, detail=f"Failed to download mmproj: {err}")


# Load vision (inference) model with mmproj
@router.post("/load")
async def load_vision_model(
    request: Request,
    data: LoadVisionInferenceRequest,
) -> LoadInferenceResponse:
    """Load a vision model with its mmproj file."""
    app: classes.FastAPIApp = request.app

    try:
        model_id = data.modelId
        model_path = data.modelPath
        mmproj_path = data.mmprojPath

        # Validate paths
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found: {model_path}")
        if not os.path.exists(mmproj_path):
            raise Exception(f"mmproj file not found: {mmproj_path}")

        # Unload existing vision model if present
        if hasattr(app.state, "vision_llm") and app.state.vision_llm:
            print(f"{common.PRNT_API} Ejecting current vision model")
            app.state.vision_llm.unload()
            app.state.vision_llm = None

        # Get model config
        model_config = get_model_install_config(model_id)
        model_name = model_config.get("model_name")

        # Create vision model instance
        app.state.vision_llm = LLAMA_CPP_VISION(
            model_path=model_path,
            mmproj_path=mmproj_path,
            model_name=model_name,
            model_id=model_id,
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
def unload_vision_model(request: Request):
    """Unload the currently loaded vision model."""
    try:
        app: classes.FastAPIApp = request.app
        if hasattr(app.state, "vision_llm") and app.state.vision_llm:
            app.state.vision_llm.unload()
        app.state.vision_llm = None
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
        # Auto-load vision model if text model is vision-capable
        if not hasattr(app.state, "vision_llm") or not app.state.vision_llm:
            # Check if text model is loaded and has vision capability
            if hasattr(app.state, "llm") and app.state.llm:
                text_llm = app.state.llm
                model_id = text_llm.model_id

                # Check if this model has an mmproj file
                mmproj_path = common.get_mmproj_path(model_id)
                model_path = common.get_model_file_path(model_id)

                if mmproj_path and model_path and os.path.exists(mmproj_path):
                    print(
                        f"{common.PRNT_API} Auto-loading vision model for {model_id}",
                        flush=True,
                    )
                    print(
                        f"{common.PRNT_API} Model path: {model_path}",
                        flush=True,
                    )
                    print(
                        f"{common.PRNT_API} mmproj path: {mmproj_path}",
                        flush=True,
                    )

                    # Get GPU layers from text model, use conservative value for vision
                    text_gpu_layers = text_llm.model_init_kwargs.get(
                        "--n-gpu-layers", 0
                    )
                    # Use fewer GPU layers for vision model to avoid memory issues
                    # Vision model runs alongside text model, so be conservative
                    vision_gpu_layers = (
                        min(text_gpu_layers, 20) if text_gpu_layers > 0 else 0
                    )

                    print(
                        f"{common.PRNT_API} Using {vision_gpu_layers} GPU layers for vision "
                        f"(text model has {text_gpu_layers})",
                        flush=True,
                    )

                    # Create init kwargs from text model's settings
                    init_kwargs = LoadTextInferenceInit(
                        n_gpu_layers=vision_gpu_layers,
                        n_ctx=text_llm.model_init_kwargs.get("--ctx-size", 4096),
                        n_batch=text_llm.model_init_kwargs.get("--batch-size", 2048),
                    )

                    # Auto-load vision model using text model's settings
                    app.state.vision_llm = LLAMA_CPP_VISION(
                        model_path=model_path,
                        mmproj_path=mmproj_path,
                        model_name=text_llm.model_name,
                        model_id=model_id,
                        verbose=True,  # Enable verbose mode for debugging
                        debug=True,  # Enable debug mode to capture stderr
                        model_init_kwargs=init_kwargs,
                        generate_kwargs=LoadTextInferenceCall(),
                    )
                    print(
                        f"{common.PRNT_API} Vision model auto-loaded successfully",
                        flush=True,
                    )
                else:
                    # Provide more detailed error message
                    error_details = []
                    if not mmproj_path:
                        error_details.append("mmproj path not found in model metadata")
                    elif not os.path.exists(mmproj_path):
                        error_details.append(f"mmproj file not found at: {mmproj_path}")
                    if not model_path:
                        error_details.append("model file path not found in metadata")
                    elif not os.path.exists(model_path):
                        error_details.append(f"model file not found at: {model_path}")

                    detail_str = (
                        "; ".join(error_details) if error_details else "unknown reason"
                    )
                    raise Exception(
                        f"No vision model loaded and text model doesn't have vision capability ({detail_str}). "
                        "Make sure the model has an mmproj file downloaded."
                    )
            else:
                raise Exception("No vision model loaded. Call /vision/load first.")

        vision_llm = app.state.vision_llm

        # Update generation settings
        vision_llm.generate_kwargs = payload

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
        response_gen = await vision_llm.vision_completion(
            prompt=payload.prompt,
            image_paths=image_paths,
            request=request,
            system_message=payload.systemMessage,
            stream=payload.stream,
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
        if hasattr(app.state, "vision_llm") and app.state.vision_llm:
            vision_llm = app.state.vision_llm
            return {
                "success": True,
                "message": f"{vision_llm.model_id} is loaded.",
                "data": {
                    "modelId": vision_llm.model_id,
                    "modelName": vision_llm.model_name,
                    "modelPath": vision_llm.model_path,
                    "mmprojPath": vision_llm.mmproj_path,
                },
            }
        else:
            return {
                "success": False,
                "message": "No vision model is currently loaded.",
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

        # Get embedding
        print(
            f"{common.PRNT_API} Creating embedding for image: {image_path}", flush=True
        )
        embeddings = await embedder.embed_images([image_path])
        embedding = embeddings[0]
        embedding_dim = len(embedding)

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

        # Prepare metadata
        source_file_name = Path(image_path).name
        metadata = {
            "type": "image",
            "source_file_name": source_file_name,
            "source_file_path": payload.image_path or "base64_upload",
            "created_at": datetime.now().isoformat(),
            "embedding_model": embedder.model_name,
            "embedding_dim": embedding_dim,
        }

        # Add transcription to metadata if available
        transcription = payload.transcription_text or None
        if payload.transcription_text:
            metadata["transcription"] = transcription

        # Store in ChromaDB if vector storage is available
        doc_id = str(uuid.uuid4())

        if hasattr(app.state, "db_client") and app.state.db_client:
            try:
                # Get or create collection
                collection = app.state.db_client.get_or_create_collection(
                    name=collection_name,
                    metadata={"type": "image_embeddings"},
                )

                # Add to collection
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    # Add raw transcription text for reference
                    documents=[transcription or ""],
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
                "embedding_dim": embedding_dim,
                "transcription": transcription,
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
def download_vision_embedding_model(payload: DownloadVisionEmbedModelRequest):
    """
    Download a vision embedding model (GGUF + mmproj) from HuggingFace.

    This downloads both the main model file and the mmproj file required
    for multimodal vision embeddings.
    """
    try:
        repo_id = payload.repo_id
        filename = payload.filename
        mmproj_filename = payload.mmproj_filename
        cache_dir = VISION_EMBEDDING_MODELS_CACHE_DIR

        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        print(
            f"{common.PRNT_API} Downloading vision embedding model: {repo_id}",
            flush=True,
        )

        # Download main model file
        print(f"{common.PRNT_API} Downloading model file: {filename}", flush=True)
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=True,
        )

        # Download mmproj file
        print(
            f"{common.PRNT_API} Downloading mmproj file: {mmproj_filename}", flush=True
        )
        hf_hub_download(
            repo_id=repo_id,
            filename=mmproj_filename,
            cache_dir=cache_dir,
            resume_download=True,
        )

        # Get actual file paths from cache
        [model_cache_info, repo_revisions] = common.scan_cached_repo(
            cache_dir=cache_dir, repo_id=repo_id
        )

        model_path = common.get_cached_blob_path(
            repo_revisions=repo_revisions, filename=filename
        )
        mmproj_path = common.get_cached_blob_path(
            repo_revisions=repo_revisions, filename=mmproj_filename
        )

        if not isinstance(model_path, str):
            raise Exception("Model path is not a string.")
        if not isinstance(mmproj_path, str):
            raise Exception("mmproj path is not a string.")

        # Calculate total size
        total_size = 0
        if os.path.exists(model_path):
            total_size += os.path.getsize(model_path)
        if os.path.exists(mmproj_path):
            total_size += os.path.getsize(mmproj_path)

        # Save metadata
        common.save_vision_embedding_model(
            repo_id=repo_id,
            model_path=model_path,
            mmproj_path=mmproj_path,
            size=total_size,
        )

        print(
            f"{common.PRNT_API} Vision embedding model downloaded successfully",
            flush=True,
        )

        return {
            "success": True,
            "message": f"Downloaded vision embedding model: {repo_id}",
            "data": {
                "repoId": repo_id,
                "modelPath": model_path,
                "mmprojPath": mmproj_path,
                "size": total_size,
            },
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
