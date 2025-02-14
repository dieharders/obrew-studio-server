import os
import json
from fastapi import APIRouter, Request, HTTPException, Depends
from .classes import RetrievalTypes
from inference.classes import (
    InferenceRequest,
    LoadedTextModelResponse,
    LoadInferenceResponse,
    LoadInferenceRequest,
    CHAT_MODES,
)
from sse_starlette.sse import EventSourceResponse

# from embeddings import main
from .llama_cpp import LLAMA_CPP
from core import classes, common
from huggingface_hub import (
    hf_hub_download,
    get_hf_file_metadata,
    hf_hub_url,
    HfApi,
)

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
        model_id = llm.model_name

        if llm:
            return {
                "success": True,
                "message": f"Model {model_id} is currently loaded.",
                "data": {
                    "modelId": model_id,
                    "mode": llm.mode,
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


# Eject the currently loaded Text Inference model
@router.post("/unload")
def unload_text_inference(request: Request):
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
        # Get the config for the model
        config_path = common.dep_path(os.path.join("public", "text_model_configs.json"))
        prompt_formats_path = common.dep_path(
            os.path.join("public", "prompt_formats.json")
        )
        with open(config_path, "r") as file:
            text_models = json.load(file)
            config = text_models[model_id]
            message_format = config["messageFormat"]
            model_name = config["name"]
        with open(prompt_formats_path, "r") as file:
            templates = json.load(file)
            message_template = templates[message_format]
        # Load the specified Ai model
        app.state.llm = LLAMA_CPP(
            model_url=None,
            model_path=modelPath,
            model_name=model_name,
            model_id=model_id,
            debug=True,  # @TODO For testing, remove when done
            mode=data.mode,
            raw=data.raw,
            message_format=message_template,
            generate_kwargs=data.call,
            model_init_kwargs=data.init,
        )
        # Init the chat conversation
        if data.mode == CHAT_MODES.CHAT.value:
            # @TODO webui needs to pass messages with a system_message as first item
            # @TODO Need chat_to_completions(chat_history) to convert conversation to string
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
        }

    # Open a new os window
    common.file_explore(filePath)

    return {
        "success": True,
        "message": "Opened file explorer",
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
        }
    except (KeyError, Exception) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


# Use Llama Index to run queries on vector database embeddings or run normal chat inference.
@router.post("/inference")
async def text_inference(
    request: Request,
    payload: InferenceRequest,
):
    app: classes.FastAPIApp = request.app
    # @TODO Re-implement this for tool use
    QUERY_INPUT = "{query_str}"
    TOOL_ARGUMENTS = "{tool_arguments_str}"
    TOOL_EXAMPLE_ARGUMENTS = "{tool_example_str}"
    TOOL_NAME = "{tool_name_str}"
    TOOL_DESCRIPTION = "{tool_description_str}"
    ASSIGNED_TOOLS = "{assigned_tools_str}"

    try:
        assigned_tool_names = payload.tools
        prompt = payload.prompt
        query_prompt = prompt
        messages = payload.messages
        collection_names = payload.collectionNames
        mode = payload.mode  # conversation type
        retrieval_type = payload.retrievalType or RetrievalTypes.BASE
        prompt_template = payload.promptTemplate
        rag_prompt_template = payload.ragPromptTemplate
        system_message = payload.systemMessage
        streaming = payload.stream

        if not app.state.llm.model_path:
            msg = "No path to model provided."
            print(f"{common.PRNT_API} Error: {msg}", flush=True)
            raise Exception(msg)
        if not app.state.llm:
            msg = "No LLM loaded."
            print(f"{common.PRNT_API} Error: {msg}", flush=True)
            raise Exception(msg)
        llm = app.state.llm

        # Update llm props
        llm.generate_kwargs = payload
        # Handles requests sequentially and streams responses using SSE
        if llm.request_queue.qsize() > 0:
            print(f"{common.PRNT_API} Too many requests, please wait.")
            return HTTPException(
                status_code=429, detail="Too many requests, please wait."
            )
        # Add request to queue
        await llm.request_queue.put(request)
        # Instruct is for Question/Answer (good for tool use, RAG)
        if mode == CHAT_MODES.INSTRUCT.value:
            # Get streaming response
            response = llm.text_completion(
                prompt=query_prompt,
                system_message=system_message,
                stream=streaming,
            )
            if streaming:
                return EventSourceResponse(response)
            # Get response
            content = [item async for item in response]
            return {
                "success": True,
                "data": content[0].get("data"),
            }
        elif mode == CHAT_MODES.CHAT.value:
            response = llm.text_chat(
                prompt=query_prompt,
                system_message=system_message,
                stream=streaming,
            )
            # Get streaming response
            if streaming:
                return EventSourceResponse(response)
            # Get response
            content = [item async for item in response]
            return {
                "success": True,
                "data": content[0].get("data"),
            }
        elif mode == CHAT_MODES.COLLAB.value:
            # @TODO Add a mode for collaborate
            raise Exception("Mode 'collab' is not implemented.")
        elif mode is None:
            raise Exception("Check 'mode' is provided.")
        else:
            raise Exception("No 'mode' or 'collection_names' provided.")
    except (KeyError, Exception) as err:
        print(f"{common.PRNT_API} Error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )
