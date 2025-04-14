from fastapi import APIRouter, Request
from core import classes
from embeddings.embedder import CHUNKING_STRATEGIES
from embeddings.response_synthesis import Response_Mode

# from llama_index.core.response_synthesizers import ResponseMode

router = APIRouter()


# Return api info for available services
@router.get("/api")
def get_services_api(request: Request) -> classes.ServicesApiResponse:
    app: classes.FastAPIApp = request.app
    data = []
    rag_response_modes = list(Response_Mode.__members__.values())

    # Return text inference services
    text_inference_api = {
        "name": "textInference",
        "port": app.state.api.SERVER_PORT,
        "configs": {
            "ragResponseModes": rag_response_modes,
        },
        "endpoints": [
            # Generate a text response from Ai engine
            {
                "name": "generate",
                "urlPath": "/v1/text/generate",
                "method": "POST",
            },
            # Stop text generation from Ai engine
            {
                "name": "stop",
                "urlPath": "/v1/text/stop",
                "method": "POST",
            },
            # Load the specified Ai model into memory
            {
                "name": "load",
                "urlPath": "/v1/text/load",
                "method": "POST",
            },
            # Eject the currently loaded Ai model from memory
            {
                "name": "unload",
                "urlPath": "/v1/text/unload",
                "method": "POST",
            },
            # Return the currently loaded model and its settings
            {
                "name": "model",
                "urlPath": "/v1/text/model",
                "method": "GET",
            },
            # Return a list of all currently installed models and their metadata
            {
                "name": "installed",
                "urlPath": "/v1/text/installed",
                "method": "GET",
            },
            # Open the directory of installed models
            {
                "name": "modelExplore",
                "urlPath": "/v1/text/modelExplore",
                "method": "GET",
            },
            # Get model install configs
            {
                "name": "getModelConfigs",
                "urlPath": "/v1/text/models",
                "method": "GET",
            },
            # Get model info
            {
                "name": "getModelInfo",
                "urlPath": "/v1/text/getModelInfo",
                "method": "GET",
            },
            # Get model details
            {
                "name": "getModelMetadata",
                "urlPath": "/v1/text/getModelMetadata",
                "method": "GET",
            },
            # Download model from Huggingface
            {
                "name": "download",
                "urlPath": "/v1/text/download",
                "method": "POST",
            },
            # Delete model from cache
            {
                "name": "delete",
                "urlPath": "/v1/text/delete",
                "method": "POST",
            },
            # llama.cpp offers native embedding too
            {
                "name": "embedding",
                "urlPath": "/v1/text/embedding",
                "method": "POST",
            },
            # Structured Data Extraction
            {
                "name": "extraction",
                "urlPath": "/v1/text/extraction",
                "method": "POST",
            },
            # Function Calling
            {
                "name": "functions",
                "urlPath": "/v1/text/functions",
                "method": "POST",
            },
            # Vision to Text (llama.cpp uses LLaVa but BakLLaVa is another option)
            {
                "name": "vision",
                "urlPath": "/v1/text/vision",
                "method": "POST",
            },
            # Code completion via Copilot
            {
                "name": "copilot",
                "urlPath": "/v1/text/copilot",
                "method": "POST",
            },
        ],
    }
    data.append(text_inference_api)

    # Return persistent file storage services
    storage_api = {
        "name": "storage",
        "port": app.state.api.SERVER_PORT,
        "endpoints": [
            # Load Chat thread
            {
                "name": "getChatThread",
                "urlPath": "/v1/persist/chat-thread",
                "method": "GET",
            },
            # Save Chat thread
            {
                "name": "saveChatThread",
                "urlPath": "/v1/persist/chat-thread",
                "method": "POST",
            },
            # Delete Chat thread
            {
                "name": "deleteChatThread",
                "urlPath": "/v1/persist/chat-thread",
                "method": "DELETE",
            },
            # Save tool setting
            {
                "name": "saveToolSettings",
                "urlPath": "/v1/persist/tool-settings",
                "method": "POST",
            },
            # Load tool settings
            {
                "name": "getToolSettings",
                "urlPath": "/v1/persist/tool-settings",
                "method": "GET",
            },
            # Delete tool settings
            {
                "name": "deleteToolSettings",
                "urlPath": "/v1/persist/tool-settings",
                "method": "DELETE",
            },
            # List tool function names
            {
                "name": "getToolFunctions",
                "urlPath": "/v1/persist/tool-funcs",
                "method": "GET",
            },
            # Get the tool function's schema for use with GUI
            {
                "name": "getToolSchema",
                "urlPath": "/v1/persist/tool-schema",
                "method": "GET",
            },
            # Load bot settings
            {
                "name": "getBotSettings",
                "urlPath": "/v1/persist/bot-settings",
                "method": "GET",
            },
            # Save bot settings
            {
                "name": "saveBotSettings",
                "urlPath": "/v1/persist/bot-settings",
                "method": "POST",
            },
            # Delete bot settings
            {
                "name": "deleteBotSettings",
                "urlPath": "/v1/persist/bot-settings",
                "method": "DELETE",
            },
        ],
    }
    data.append(storage_api)

    # Return search over local files services
    memory_api = {
        "name": "memory",
        "port": app.state.api.SERVER_PORT,
        "configs": {
            "chunkingStrategies": list(CHUNKING_STRATEGIES.keys()),
        },
        "endpoints": [
            {
                "name": "addCollection",
                "urlPath": "/v1/memory/addCollection",
                "method": "GET",
            },
            {
                "name": "getCollection",
                "urlPath": "/v1/memory/getCollection",
                "method": "POST",
            },
            {
                "name": "getAllCollections",
                "urlPath": "/v1/memory/getAllCollections",
                "method": "GET",
            },
            {
                "name": "deleteCollection",
                "urlPath": "/v1/memory/deleteCollection",
                "method": "GET",
            },
            {
                "name": "addDocument",
                "urlPath": "/v1/memory/addDocument",
                "method": "POST",
            },
            {
                "name": "getChunks",
                "urlPath": "/v1/memory/getChunks",
                "method": "POST",
            },
            {
                "name": "deleteDocuments",
                "urlPath": "/v1/memory/deleteDocuments",
                "method": "POST",
            },
            {
                "name": "fileExplore",
                "urlPath": "/v1/memory/fileExplore",
                "method": "GET",
            },
            {
                "name": "updateDocument",
                "urlPath": "/v1/memory/updateDocument",
                "method": "POST",
            },
            {
                "name": "wipe",
                "urlPath": "/v1/memory/wipe",
                "method": "GET",
            },
        ],
    }
    data.append(memory_api)

    return {
        "success": True,
        "message": "These are the currently available service api's",
        "data": data,
    }
