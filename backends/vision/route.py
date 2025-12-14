import os
import json
import tempfile
from fastapi import APIRouter, Request, HTTPException
from .llama_cpp_vision import LLAMA_CPP_VISION
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
)
from inference.helpers import decode_base64_image, cleanup_temp_images


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


# Load vision model with mmproj
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


# Unload vision model
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
            image_paths.append(path)

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
        print(f"{common.PRNT_API} Vision generation error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Vision generation failed: {err}",
            "data": None,
        }
    finally:
        # Cleanup temporary images
        if temp_image_paths:
            cleanup_temp_images(temp_image_paths)


# Get currently loaded vision model info
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
