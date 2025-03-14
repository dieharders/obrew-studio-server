import os
import json
from fastapi import APIRouter, Request, HTTPException, Depends
from inference.agent import Agent
from .llama_cpp import LLAMA_CPP
from core import classes, common
from huggingface_hub import (
    hf_hub_download,
    get_hf_file_metadata,
    hf_hub_url,
    HfApi,
)
from inference.classes import (
    AgentOutput,
    InferenceRequest,
    LoadedTextModelResponse,
    LoadInferenceResponse,
    LoadInferenceRequest,
    CHAT_MODES,
)


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
            return dict(
                message_format=message_format, model_name=model_name, models=text_models
            )
    except Exception as err:
        raise Exception(f"Error finding models list: {err}")


def get_prompt_formats(message_format: str) -> dict:
    try:
        # Get the file for the templates
        prompt_formats_path = common.dep_path(
            os.path.join("public", "prompt_formats.json")
        )
        with open(prompt_formats_path, "r") as file:
            templates = json.load(file)
            message_template = templates[message_format]
            return message_template
    except Exception as err:
        raise Exception(f"Error finding prompt format templates: {err}")


router = APIRouter()


# Return a list of all currently installed models and their metadata
@router.get("/installed")
def get_installed_models() -> classes.TextModelInstallMetadataResponse:
    try:
        data = []
        # Get installed models file
        metadatas = common.get_settings_file(
            common.APP_SETTINGS_PATH, common.MODEL_METADATAS_FILEPATH
        )
        if not metadatas:
            metadatas = common.DEFAULT_SETTINGS_DICT
        if common.INSTALLED_TEXT_MODELS in metadatas:
            data = metadatas[common.INSTALLED_TEXT_MODELS]
            return {
                "success": True,
                "message": "This is a list of all currently installed models.",
                "data": data,
            }
        else:
            raise Exception(
                f"No attribute {common.INSTALLED_TEXT_MODELS} exists in settings file."
            )
    except Exception as err:
        return {
            "success": False,
            "message": f"Failed to find any installed models. {err}",
            "data": [],
        }


# Gets the currently loaded model and its installation/config metadata
@router.get("/model")
def get_text_model(request: Request) -> LoadedTextModelResponse | dict:
    app: classes.FastAPIApp = request.app

    try:
        llm = app.state.llm
        model_id = llm.model_id

        if llm:
            return {
                "success": True,
                "message": f"{model_id} is loaded.",
                "data": {
                    "modelId": model_id,
                    "modelName": llm.model_name,
                    "responseMode": llm.response_mode,
                    "modelSettings": llm.model_init_kwargs,
                    "generateSettings": llm.generate_kwargs,
                },
            }
        else:
            return {
                "success": False,
                "message": "No model is currently loaded.",
                "data": {},
            }
    except (Exception, KeyError, HTTPException) as error:
        return {
            "success": False,
            "message": f"Something went wrong: {error}",
            "data": {},
        }


# Returns the curated list of models available for installation from json file
@router.get("/models")
def get_model_list():
    try:
        #  Get data from file
        file = get_model_install_config()
        models_list = file["models"]

        return {
            "success": True,
            "message": "This is the curated list of models for download.",
            "data": models_list,
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Something went wrong. Reason: {err}",
            "data": None,
        }


# Eject the currently loaded Text Inference model
@router.post("/unload")
def unload_text_inference(request: Request):
    try:
        app: classes.FastAPIApp = request.app
        if app.state.llm:
            app.state.llm.unload()
        del app.state.llm
        app.state.llm = None
        return {
            "success": True,
            "message": "Model was ejected",
            "data": None,
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Error ejecting model. Reason: {err}",
            "data": None,
        }


# Start Text Inference service
@router.post("/load")
async def load_text_inference(
    request: Request,
    data: LoadInferenceRequest,
) -> LoadInferenceResponse:
    app: classes.FastAPIApp = request.app

    try:
        model_id = data.modelId
        modelPath = data.modelPath
        # Unload the model if it exists
        if app.state.llm:
            print(
                f"{common.PRNT_API} Ejecting current model {model_id} from: {modelPath}"
            )
            unload_text_inference(request)
        # Load the config for the model
        model_config = get_model_install_config(model_id)
        message_format_id = model_config["message_format"]
        model_name = model_config["model_name"]
        # Load the prompt formats
        message_template = get_prompt_formats(message_format_id)
        # Load the specified Ai model using a specific inference backend
        app.state.llm = LLAMA_CPP(
            model_url=None,
            model_path=modelPath,
            model_name=model_name,
            model_id=model_id,
            # debug=True,  # For testing
            response_mode=data.responseMode,
            active_role=data.activeRole,
            raw=data.raw,
            message_format=message_template,
            generate_kwargs=data.call,
            model_init_kwargs=data.init,
        )
        if data.responseMode == CHAT_MODES.CHAT.value:
            # @TODO webui needs to pass messages list with a system_message as first msg
            await app.state.llm.load_chat(chat_history=data.messages)
        # Return result
        print(f"{common.PRNT_API} Model {model_id} loaded from: {modelPath}")
        return {
            "message": f"AI model [{model_id}] loaded.",
            "success": True,
            "data": None,
        }
    except (Exception, KeyError) as error:
        return {
            "message": f"Unable to load AI model [{model_id}]\nMake sure you have available system memory.\n{error}",
            "success": False,
            "data": None,
        }
    except (FileNotFoundError, json.JSONDecodeError) as error:
        return {
            "message": f"Unable to load AI model [{model_id}]\nError: Invalid JSON format or file not found.\n{error}",
            "success": False,
            "data": None,
        }


# Open OS file explorer on host machine
@router.get("/modelExplore")
def explore_text_model_dir() -> classes.FileExploreResponse:
    filePath = common.app_path(common.TEXT_MODELS_CACHE_DIR)

    if not os.path.exists(filePath):
        return {
            "success": False,
            "message": "No file path exists",
            "data": None,
        }

    # Open a new os window
    common.file_explore(filePath)

    return {
        "success": True,
        "message": "Opened file explorer",
        "data": None,
    }


# @TODO Search huggingface hub and return results
# https://huggingface.co/docs/huggingface_hub/en/guides/search
@router.get("/searchModels")
def search_models(payload):
    sort = payload.sort
    task = payload.task or "text-generation"
    limit = payload.limit or 10
    hf_api = HfApi()
    # Example showing how to filter by task and return only top 10 most downloaded
    models = hf_api.list_models(
        sort=sort,  # or "downloads" or "trending"
        limit=limit,
        task=task,
    )
    return {
        "success": True,
        "message": f"Returned {len(models)} results",
        "data": models,
    }


# Fetches repo info about a model from huggingface hub
@router.get("/getModelInfo")
def get_model_info(
    payload: classes.GetModelInfoRequest = Depends(),
):
    id = payload.repoId
    hf_api = HfApi()
    info = hf_api.model_info(repo_id=id, files_metadata=True)
    return {
        "success": True,
        "message": "Returned model info",
        "data": info,
    }


# Fetches metadata about a file from huggingface hub
@router.get("/getModelMetadata")
def get_model_metadata(payload):
    repo_id = payload.repo_id
    filename = payload.filename
    url = hf_hub_url(repo_id=repo_id, filename=filename)
    metadata = get_hf_file_metadata(url=url)

    return {
        "success": True,
        "message": "Returned model metadata",
        "data": metadata,
    }


# Download a text model from huggingface hub
# https://huggingface.co/docs/huggingface_hub/v0.21.4/en/package_reference/file_download#huggingface_hub.hf_hub_download
@router.post("/download")
def download_text_model(payload: classes.DownloadTextModelRequest):
    try:
        repo_id = payload.repo_id
        filename = payload.filename
        cache_dir = common.app_path(common.TEXT_MODELS_CACHE_DIR)
        resume_download = False
        # repo_type = "model" # optional, specify type of data, defaults to model
        # local_dir = "" # optional, downloaded file will be placed under this directory

        # Save initial path and details to json file
        common.save_text_model(
            {
                "repoId": repo_id,
                "savePath": {filename: ""},
            }
        )

        # Download model.
        # Returned path is symlink which isnt loadable; for our purposes we use get_cached_blob_path().
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            resume_download=resume_download,
            # local_dir=cache_dir,
            # local_dir_use_symlinks=False,
            # repo_type=repo_type,
        )

        # Get actual file path
        [model_cache_info, repo_revisions] = common.scan_cached_repo(
            cache_dir=cache_dir, repo_id=repo_id
        )
        # Get from dl path
        # file_path = common.app_path(download_path)

        # Get from huggingface hub managed cache dir
        file_path = common.get_cached_blob_path(
            repo_revisions=repo_revisions, filename=filename
        )
        if not isinstance(file_path, str):
            raise Exception("Path is not string.")

        # Save finalized details to disk
        common.save_text_model(
            {
                "repoId": repo_id,
                "savePath": {filename: file_path},
            }
        )

        return {
            "success": True,
            "message": f"Saved model file to {file_path}.",
            "data": None,
        }
    except (KeyError, Exception, EnvironmentError, OSError, ValueError) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


# Remove text model weights file and installation record.
# Current limitation is that this deletes all quant files for a repo.
@router.post("/delete")
def delete_text_model(payload: classes.DeleteTextModelRequest):
    filename = payload.filename
    repo_id = payload.repoId

    try:
        cache_dir = common.app_path(common.TEXT_MODELS_CACHE_DIR)

        # Checks file and throws if not found
        common.check_cached_file_exists(
            cache_dir=cache_dir, repo_id=repo_id, filename=filename
        )

        # Find model hash
        [model_cache_info, repo_revisions] = common.scan_cached_repo(
            cache_dir=cache_dir, repo_id=repo_id
        )
        repo_commit_hash = []
        for r in repo_revisions:
            repo_commit_hash.append(r.commit_hash)

        # Delete weights from cache, https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
        delete_strategy = model_cache_info.delete_revisions(*repo_commit_hash)
        delete_strategy.execute()
        freed_size = delete_strategy.expected_freed_size_str
        print(f"{common.PRNT_API} Freed {freed_size} space.", flush=True)

        # Delete install record from json file
        if freed_size != "0.0":
            common.delete_text_model_revisions(repo_id=repo_id)

        return {
            "success": True,
            "message": f"Deleted model file from {filename}. Freed {freed_size} of space.",
            "data": None,
        }
    except (KeyError, Exception) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


# Run text inference
@router.post("/generate")
async def generate_text(
    request: Request,
    payload: InferenceRequest,
) -> AgentOutput | classes.GenericEmptyResponse:
    app: classes.FastAPIApp = request.app

    try:
        response_type = payload.responseMode  # conversation type
        streaming = payload.stream
        system_message = payload.systemMessage
        prompt = payload.prompt
        prompt_template = payload.promptTemplate
        assigned_tool_names = payload.tools
        # messages = payload.messages # @TODO Implement...

        # We can re-use llm for multi-turn conversations
        llm = app.state.llm

        # Update llm props
        llm.generate_kwargs = payload

        # Verify there is a model to run
        if not llm.model_path:
            msg = "No path to model provided."
            print(f"{common.PRNT_API} Error: {msg}", flush=True)
            raise Exception(msg)
        if not llm:
            msg = "No LLM loaded."
            print(f"{common.PRNT_API} Error: {msg}", flush=True)
            raise Exception(msg)

        # Assign Agent
        agent = Agent(llm=llm, tools=assigned_tool_names, active_role=llm.active_role)
        response = await agent.call(
            request=request,
            system_message=system_message,
            prompt=prompt,
            prompt_template=prompt_template,
            streaming=streaming,
            response_type=response_type,
        )
        # Cleanup/complete request
        await llm.complete_request()
        # Return final answer
        return response
    except (KeyError, Exception) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        # Cleanup/complete request
        await llm.complete_request()
        return {
            "success": False,
            "message": f"Something went wrong. Reason: {err}",
            "data": None,
        }
