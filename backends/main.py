import os
import re
import glob
import json
import uvicorn
import subprocess
import httpx
import shutil
from typing import List, Tuple, Optional
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    File,
    UploadFile,
    Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from inference import text_llama_index, text_llama_cpp_python, text_routes
from embedding import embedding

FILEBROWSER_PATH = os.path.join(os.getenv("WINDIR"), "explorer.exe")
VECTOR_DB_FOLDER = "chromadb"
VECTOR_STORAGE_PATH = os.path.join(os.getcwd(), VECTOR_DB_FOLDER)
MEMORY_FOLDER = "memories"
PARSED_FOLDER = "parsed"
TMP_FOLDER = "tmp"
MEMORY_PATH = os.path.join(os.getcwd(), MEMORY_FOLDER)
PARSED_DOCUMENT_PATH = os.path.join(MEMORY_PATH, PARSED_FOLDER)
TMP_DOCUMENT_PATH = os.path.join(MEMORY_PATH, TMP_FOLDER)


@asynccontextmanager
async def lifespan(application: FastAPI):
    print("[homebrew api] Lifespan startup")
    # https://www.python-httpx.org/quickstart/
    app.requests_client = httpx.Client()
    # Store some state here if you want...
    app.text_inference_process = None
    application.state.storage_directory = VECTOR_STORAGE_PATH
    application.state.db_client = None
    application.state.llm = None  # Set each time user loads a model
    application.state.path_to_model = ""  # Set each time user loads a model
    application.state.text_model_config = None

    yield

    print("[homebrew api] Lifespan shutdown")
    kill_text_inference()


app = FastAPI(title="🍺 HomeBrew API server", version="0.1.0", lifespan=lifespan)


# Configure CORS settings
origins = [
    "http://localhost:3000",  # (optional) for testing client apps
    "https://hoppscotch.io",  # (optional) for testing endpoints
    "http://localhost:8000",  # (required) Homebrew front-end
    "https://brain-dump-dieharders.vercel.app",  # (required) client app origin (preview)
    "https://homebrew-ai-discover.vercel.app",  # (required) client app origin (production)
]


# Redirect requests to our custom endpoints
# from fastapi import Request
# @app.middleware("http")
# async def redirect_middleware(request: Request, call_next):
#     return await redirects.text(request, call_next, str(app.PORT_TEXT_INFERENCE))


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PingResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "pong",
                }
            ]
        }
    }


# Keep server/database alive
@app.get("/v1/ping")
def ping() -> PingResponse:
    try:
        db = get_vectordb_client()
        db.heartbeat()
        return {"success": True, "message": "pong"}
    except Exception as e:
        print(f"[homebrew api] Error pinging server: {e}")
        return {"success": False, "message": ""}


class ConnectResponse(BaseModel):
    success: bool
    message: str
    data: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Connected to api server on port 8008.",
                    "data": {"docs": "http://localhost:8008/docs"},
                }
            ]
        }
    }


# Tell client we are ready to accept requests
@app.get("/v1/connect")
def connect() -> ConnectResponse:
    return {
        "success": True,
        "message": f"Connected to api server on port {app.PORT_HOMEBREW_API}. Refer to 'http://localhost:{app.PORT_HOMEBREW_API}/docs' for api docs.",
        "data": {
            "docs": f"http://localhost:{app.PORT_HOMEBREW_API}/docs",
        },
    }


# Load in the ai model to be used for inference.
class LoadInferenceRequest(BaseModel):
    modelId: str
    pathToModel: str
    textModelConfig: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "modelId": "llama-2-13b-chat-ggml",
                    "pathToModel": "C:\\homebrewai-app\\models\\llama-2-13b.GGUF",
                    "textModelConfig": {
                        "promptTemplate": "Instructions:{{PROMPT}}\n\n### Response:",
                        "savePath": "C:\\Project Files\\brain-dump-ai\\models\\llama-2-13b-chat.ggmlv3.q2_K.bin",
                        "id": "llama2-13b",
                        "numTimesRun": 0,
                        "isFavorited": False,
                        "validation": "success",
                        "modified": "Tue, 19 Sep 2023 23:25:28 GMT",
                        "size": 1200000,
                        "endChunk": 13,
                        "progress": 67,
                        "tokenizerPath": "/some/path/to/tokenizer",
                        "checksum": "90b27795b2e319a93cc7c3b1a928eefedf7bd6acd3ecdbd006805f7a028ce79d",
                    },
                }
            ]
        }
    }


class LoadInferenceResponse(BaseModel):
    message: str
    success: bool

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "AI model [llama-2-13b-chat-ggml] loaded.",
                    "success": True,
                }
            ]
        }
    }


@app.post("/v1/text/load")
def load_text_inference(data: LoadInferenceRequest) -> LoadInferenceResponse:
    try:
        # Store the current model's configuration for later reference
        app.state.text_model_config = data.textModelConfig
        model_id = data.modelId
        app.state.path_to_model = data.pathToModel
        print(f"[homebrew api] Path to model loaded: {data.pathToModel}")
        # Logic to load the specified ai model here...
        return {"message": f"AI model [{model_id}] loaded.", "success": True}
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid JSON format: missing key")


class StartInferenceRequest(BaseModel):
    modelConfig: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "modelConfig": {
                        "promptTemplate": "Instructions:{{PROMPT}}\n\n### Response:",
                        "savePath": "C:\\Project Files\\brain-dump-ai\\models\\llama-2-13b-chat.ggmlv3.q2_K.bin",
                        "id": "llama2-13b",
                        "numTimesRun": 0,
                        "isFavorited": False,
                        "validation": "success",
                        "modified": "Tue, 19 Sep 2023 23:25:28 GMT",
                        "size": 1200000,
                        "endChunk": 13,
                        "progress": 67,
                        "tokenizerPath": "/some/path/to/tokenizer",
                        "checksum": "90b27795b2e319a93cc7c3b1a928eefedf7bd6acd3ecdbd006805f7a028ce79d",
                    },
                }
            ]
        }
    }


class StartInferenceResponse(BaseModel):
    success: bool
    message: str
    data: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "AI text inference started.",
                    "data": {
                        "port": 8080,
                        "docs": "http://localhost:8080/docs",
                        "textModelConfig": {
                            "promptTemplate": "Instructions:{{PROMPT}}\n\n### Response:",
                            "savePath": "C:\\Project Files\\brain-dump-ai\\models\\llama-2-13b-chat.ggmlv3.q2_K.bin",
                            "id": "llama2-13b",
                            "numTimesRun": 0,
                            "isFavorited": False,
                            "validation": "success",
                            "modified": "Tue, 19 Sep 2023 23:25:28 GMT",
                            "size": 1200000,
                            "endChunk": 13,
                            "progress": 67,
                            "tokenizerPath": "/some/path/to/tokenizer",
                            "checksum": "90b27795b2e319a93cc7c3b1a928eefedf7bd6acd3ecdbd006805f7a028ce79d",
                        },
                    },
                }
            ]
        }
    }


# Starts the text inference server
@app.post("/v1/text/start")
async def start_text_inference(data: StartInferenceRequest) -> StartInferenceResponse:
    try:
        # Store the current model's configuration for later reference
        app.state.text_model_config = data.modelConfig
        # Send signal to start server
        model_file_path: str = data.modelConfig["savePath"]
        app.text_inference_process = (
            await text_llama_cpp_python.start_text_inference_server(
                model_file_path,
                app.PORT_TEXT_INFERENCE,
            )
        )

        if app.text_inference_process:
            isStarted = True
        else:
            isStarted = False

        return {
            "success": isStarted,
            "message": "AI inference started.",
            "data": {
                "port": app.PORT_TEXT_INFERENCE,
                "docs": f"http://localhost:{app.PORT_TEXT_INFERENCE}/docs",
                "text_model_config": data.modelConfig,
            },
        }
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON format: 'modelConfig' key not found",
        )


class ShutdownInferenceResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"message": "Services shutdown.", "success": True}]
        }
    }


# Shutdown all currently open processes/subprocesses for text inferencing.
@app.get("/v1/services/shutdown")
async def shutdown_text_inference() -> ShutdownInferenceResponse:
    try:
        print("[homebrew api] Shutting down all services")
        # Reset, kill processes
        kill_text_inference()
        delattr(app.state, "text_model_config")

        return {
            "success": True,
            "message": "Services shutdown successfully.",
        }
    except Exception as e:
        print(f"[homebrew api] Error shutting down services: {e}")
        return {
            "success": False,
            "message": f"Error shutting down services: {e}",
        }


class ServicesApiResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "These are api params for accessing services endpoints.",
                    "data": [
                        {
                            "name": "textInference",
                            "port": 8008,
                            "endpoints": [
                                {
                                    "name": "completions",
                                    "urlPath": "/v1/completions",
                                    "method": "POST",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    }


# Return api info for available services
@app.get("/v1/services/api")
def get_services_api() -> ServicesApiResponse:
    data = []

    # Only return api configs for servers that are actually running
    # if hasattr(app, "text_model_config"):
    #     text_inference_api = {
    #         "name": "textInference",
    #         "port": app.PORT_TEXT_INFERENCE,
    #         "endpoints": [
    #             {
    #                 "name": "copilot",
    #                 "urlPath": "/v1/engines/copilot-codex/completions",
    #                 "method": "POST",
    #             },
    #             {
    #                 "name": "completions",
    #                 "urlPath": "/v1/completions",
    #                 "method": "POST",
    #                 "promptTemplate": app.text_model_config["promptTemplate"],
    #             },
    #             {"name": "embeddings", "urlPath": "/v1/embeddings", "method": "POST"},
    #             {
    #                 "name": "chatCompletions",
    #                 "urlPath": "/v1/chat/completions",
    #                 "method": "POST",
    #             },
    #             {"name": "models", "urlPath": "/v1/models", "method": "GET"},
    #         ],
    #     }
    #     data.append(text_inference_api)

    # Return text inference services available from Homebrew
    text_inference_api = {
        "name": "textInference",
        "port": app.PORT_HOMEBREW_API,
        "endpoints": [
            {
                "name": "inference",
                "urlPath": "/v1/text/inference",
                "method": "POST",
                "promptTemplate": app.state.text_model_config["promptTemplate"],
            },
        ],
    }
    data.append(text_inference_api)

    # Return services that are ready now
    memory_api = {
        "name": "memory",
        "port": app.PORT_HOMEBREW_API,
        "endpoints": [
            {
                "name": "addCollection",
                "urlPath": "/v1/memory/addCollection",
                "method": "GET",
            },
            {
                "name": "getAllCollections",
                "urlPath": "/v1/memory/getAllCollections",
                "method": "GET",
            },
            {
                "name": "getCollection",
                "urlPath": "/v1/memory/getCollection",
                "method": "POST",
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
                "name": "getDocument",
                "urlPath": "/v1/memory/getDocument",
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


class InferenceRequest(BaseModel):
    prompt: str
    collectionNames: Optional[List[str]] = []
    mode: Optional[str] = "completion"

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Why does mass conservation break down?",
                    "collectionNames": ["science"],
                    "mode": "completion",
                }
            ]
        }
    }


class InferenceResponse(BaseModel):
    success: bool
    message: str
    data: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Successfull text inference",
                    "success": True,
                    "data": "This is a response.",
                }
            ]
        }
    }


# Use Llama Index to run queries on vector database embeddings.
@app.post("/v1/text/inference")
async def text_inference(payload: InferenceRequest):
    try:
        prompt = payload.prompt
        collection_names = payload.collectionNames
        mode = payload.mode

        print(
            f"[homebrew api] text_inference: {prompt} on: {collection_names} in mode {mode}"
        )

        if not app.state.path_to_model:
            raise Exception("No path to model provided.")
        if not app.state.text_model_config:
            raise Exception("No model config exists.")

        # Call LLM
        if len(collection_names):
            return EventSourceResponse(
                token_streamer(query_memory(prompt, collection_names)),
            )
        else:
            # @TODO Return the same streaming event response as query using llamaIndex
            result = text_routes.inference_completions(prompt)
            return {
                "message": "Inference complete",
                "success": False,
                "data": result,
            }
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid JSON format: missing key")


class PreProcessRequest(BaseModel):
    document_id: Optional[str] = ""
    document_name: str
    collection_name: str
    description: Optional[str] = ""
    tags: Optional[str] = ""
    filePath: str


class PreProcessResponse(BaseModel):
    success: bool
    message: str
    data: dict[str, str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Successfully processed file",
                    "success": True,
                    "data": {
                        "document_id": "1010-1010",
                        "file_name": "filename.md",
                        "path_to_file": "C:\\app_data\\parsed",
                        "checksum": "xxx",
                    },
                }
            ]
        }
    }


# Pre-process supplied files into a text format and save to disk for embedding later.
@app.post("/v1/embeddings/preProcess")
def pre_process_documents(form: PreProcessRequest = Depends()) -> PreProcessResponse:
    try:
        # Validate inputs
        document_id = form.document_id
        file_path = form.filePath
        collection_name = form.collection_name
        document_name = form.document_name
        if not check_valid_id(collection_name) or not check_valid_id(document_name):
            raise Exception(
                "Invalid input. No '--', uppercase, spaces or special chars allowed."
            )
        # Validate tags
        parsed_tags = parse_valid_tags(form.tags)
        if parsed_tags == None:
            raise Exception("Invalid value for 'tags' input.")
        # Process files
        processed_file = embedding.pre_process_documents(
            document_id=document_id,
            document_name=document_name,
            collection_name=collection_name,
            description=form.description,
            tags=parsed_tags,
            input_file_path=file_path,
            output_folder_path=PARSED_DOCUMENT_PATH,
        )

        return {
            "success": True,
            "message": f"Successfully processed {file_path}",
            "data": processed_file,
        }
    except (Exception, ValueError, TypeError, KeyError) as error:
        return {
            "success": False,
            "message": f"There was an internal server error uploading the file:\n{error}",
        }


class AddCollectionRequest(BaseModel):
    collectionName: str
    description: Optional[str] = ""
    tags: Optional[str] = List[None]


class AddCollectionResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Successfully created new collection",
                    "success": True,
                }
            ]
        }
    }


@app.get("/v1/memory/addCollection")
def create_memory_collection(
    form: AddCollectionRequest = Depends(),
) -> AddCollectionResponse:
    try:
        parsed_tags = parse_valid_tags(form.tags)
        collection_name = form.collectionName
        if not collection_name:
            raise Exception("You must supply a collection name.")
        if parsed_tags == None:
            raise Exception("Invalid value for 'tags' input.")
        if not check_valid_id(collection_name):
            raise Exception(
                "Invalid collection name. No '--', uppercase, spaces or special chars allowed."
            )
        # Create payload. ChromaDB only accepts strings, numbers, bools.
        metadata = {
            "tags": parsed_tags,
            "description": form.description,
            "sources": json.dumps([]),
        }
        # Apply input values to collection metadata
        db_client = get_vectordb_client()
        db_client.create_collection(
            name=collection_name,
            metadata=metadata,
        )
        return {
            "success": True,
            "message": f"Successfully created new collection [{collection_name}]",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to create new collection [{collection_name}]: {e}",
        }


class AddDocumentRequest(BaseModel):
    documentName: str
    collectionName: str
    description: Optional[str] = ""
    tags: Optional[str] = ""
    urlPath: Optional[str] = ""


class AddDocumentResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "A new memory has been added",
                    "success": True,
                }
            ]
        }
    }


# Create a memory for Ai.
# This is a multi-step process involving several endpoints.
# It will first process the file, then embed its data into vector space and
# finally add it as a document to specified collection.
@app.post("/v1/memory/addDocument")
async def create_memory(
    form: AddDocumentRequest = Depends(),
    file: UploadFile = File(None),  # File(...) means required
    background_tasks: BackgroundTasks = None,  # This prop is auto populated by FastAPI
) -> AddDocumentResponse:
    try:
        document_name = form.documentName
        collection_name = form.collectionName
        description = form.description
        url_path = form.urlPath
        tags = parse_valid_tags(form.tags)
        tmp_input_file_path = ""

        if file == None and url_path == "":
            raise Exception("You must supply a file upload or url.")
        if not document_name or not collection_name:
            raise Exception("You must supply a collection and memory name.")
        if tags == None:
            raise Exception("Invalid value for 'tags' input.")
        if not check_valid_id(document_name):
            raise Exception(
                "Invalid memory name. No '--', uppercase, spaces or special chars allowed."
            )
        if not app.state.path_to_model:
            raise Exception("No model path defined.")

        # Save temp files to disk first. The filename doesnt matter much.
        tmp_folder = TMP_DOCUMENT_PATH
        filename = embedding.create_parsed_filename(collection_name, document_name)
        tmp_input_file_path = os.path.join(tmp_folder, filename)
        if url_path:
            print(
                f"[homebrew api] Downloading file from url {url_path} to {tmp_input_file_path}"
            )
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            # Download the file and save to disk
            await get_file_from_url(url_path, tmp_input_file_path)
        elif file:
            print("[homebrew api] Saving uploaded file to disk...")
            # Read the uploaded file in chunks of 1mb,
            # store to a tmp dir for processing later
            if not os.path.exists(tmp_folder):
                os.makedirs(tmp_folder)
            with open(tmp_input_file_path, "wb") as f:
                while contents := file.file.read(1024 * 1024):
                    f.write(contents)
            file.file.close()
        else:
            raise Exception("No file or url supplied")

        # Parse/Process input files
        processed_file = embedding.pre_process_documents(
            document_name=document_name,
            collection_name=collection_name,
            description=description,
            tags=tags,
            input_file_path=tmp_input_file_path,
            output_folder_path=PARSED_DOCUMENT_PATH,
        )

        # Create embeddings
        print("[homebrew api] Start embedding...")
        if app.state.llm == None:
            app.state.llm = text_llama_index.load_text_model(app.state.path_to_model)
        db_client = get_vectordb_client()
        embed_form = {
            "collection_name": collection_name,
            "document_name": document_name,
            "document_id": processed_file["document_id"],
            "description": description,
            "tags": tags,
            "is_update": False,
        }
        background_tasks.add_task(
            embedding.create_embedding,
            processed_file,
            app.state.storage_directory,
            embed_form,
            app.state.llm,
            db_client,
        )
    except (Exception, KeyError) as e:
        # Error
        msg = f"Failed to create a new memory: {e}"
        print(f"[homebrew api] {msg}")
        return {
            "success": False,
            "message": msg,
        }
    else:
        msg = "A new memory has been added to the queue. It will be available for use shortly."
        print(f"[homebrew api] {msg}")
        return {
            "success": True,
            "message": msg,
        }
    finally:
        # Delete uploaded tmp file
        if os.path.exists(tmp_input_file_path):
            os.remove(tmp_input_file_path)
            print(f"[homebrew api] Removed temp file.")


class GetAllCollectionsResponse(BaseModel):
    success: bool
    message: str
    data: list

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Returned 5 collection(s)",
                    "success": True,
                    "data": [
                        {
                            "name": "collection-name",
                            "id": "1010-10101",
                            "metadata": {
                                "description": "A description.",
                                "sources": ["document-id"],
                                "tags": "html5 react",
                            },
                        }
                    ],
                }
            ]
        }
    }


@app.get("/v1/memory/getAllCollections")
def get_all_collections() -> GetAllCollectionsResponse:
    try:
        db = get_vectordb_client()
        collections = db.list_collections()

        # Parse json data
        for collection in collections:
            metadata = collection.metadata
            if "sources" in metadata:
                sources_json = metadata["sources"]
                sources_data = json.loads(sources_json)
                metadata["sources"] = sources_data

        return {
            "success": True,
            "message": f"Returned {len(collections)} collection(s)",
            "data": collections,
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": e,
        }


class GetCollectionRequest(BaseModel):
    id: str
    include: Optional[List[str]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "examples",
                    "include": ["embeddings", "documents"],
                }
            ]
        }
    }


class GetCollectionResponse(BaseModel):
    success: bool
    message: str
    data: dict

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Returned 5 source(s) in collection",
                    "success": True,
                    "data": {
                        "collection": {},
                        "numItems": 5,
                    },
                }
            ]
        }
    }


# Return a collection by id and all its documents
@app.post("/v1/memory/getCollection")
def get_collection(props: GetCollectionRequest) -> GetCollectionResponse:
    try:
        db = get_vectordb_client()
        id = props.id
        collection = db.get_collection(id)
        num_items = 0
        metadata = collection.metadata
        if metadata == None:
            raise Exception("No metadata found for collection")

        if "sources" in metadata:
            sources_json = metadata["sources"]
            sources_data = json.loads(sources_json)
            num_items = len(sources_data)
            metadata["sources"] = sources_data

        return {
            "success": True,
            "message": f"Returned {num_items} source(s) in collection [{id}]",
            "data": {
                "collection": collection,
                "numItems": num_items,
            },
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": e,
        }


class GetDocumentRequest(BaseModel):
    collection_id: str
    document_ids: List[str]
    include: Optional[List[str]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_id": "examples",
                    "document_ids": ["science"],
                    "include": ["embeddings", "documents"],
                }
            ]
        }
    }


class GetDocumentResponse(BaseModel):
    success: bool
    message: str
    data: List[dict]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Returned 1 document(s)",
                    "success": True,
                    "data": [{}],
                }
            ]
        }
    }


# Get one or more documents by id.
@app.post("/v1/memory/getDocument")
def get_document(params: GetDocumentRequest) -> GetDocumentResponse:
    try:
        collection_id = params.collection_id
        document_ids = params.document_ids
        include = params.include

        documents = embedding.get_document(
            collection_name=collection_id,
            document_ids=document_ids,
            db=get_vectordb_client(),
            include=include,
        )

        num_documents = len(documents)

        return {
            "success": True,
            "message": f"Returned {num_documents} document(s)",
            "data": documents,
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": e,
        }


class ExploreSourceRequest(BaseModel):
    filePath: str


class FileExploreResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Opened file explorer",
                    "success": True,
                }
            ]
        }
    }


# Open an OS file exporer on host machine
@app.get("/v1/memory/fileExplore")
def explore_source_file(
    params: ExploreSourceRequest = Depends(),
) -> FileExploreResponse:
    filePath = params.filePath

    if not filePath:
        return {
            "success": False,
            "message": "No file path given",
        }
    # Open a new os window
    file_explore(filePath)

    return {
        "success": True,
        "message": "Opened file explorer",
    }


class UpdateDocumentRequest(BaseModel):
    collectionName: str
    documentName: str
    documentId: str
    urlPath: Optional[str] = ""
    filePath: Optional[str] = ""
    metadata: Optional[dict] = {}


class UpdateDocumentResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Updated memories",
                    "success": True,
                }
            ]
        }
    }


# Re-process and re-embed existing document(s) from /parsed directory or url link
@app.post("/v1/memory/updateDocument")
async def update_memory(
    args: UpdateDocumentRequest,
    background_tasks: BackgroundTasks = None,  # This prop is auto populated by FastAPI
) -> UpdateDocumentResponse:
    try:
        collection_name = args.collectionName
        document_id = args.documentId
        document_name = args.documentName
        metadata = args.metadata
        url_path = args.urlPath
        file_path = args.filePath
        document = None
        document_metadata = {}

        # Verify id's
        if not collection_name or not document_name or not document_id:
            raise Exception(
                "Please supply a collection name, document name, and document id"
            )
        if not check_valid_id(document_name):
            raise Exception(
                "Invalid memory name. No '--', uppercase, spaces or special chars allowed."
            )

        # Retrieve document data
        db = get_vectordb_client()
        documents = embedding.get_document(
            collection_name=collection_name,
            document_ids=[document_id],
            db=db,
            include=["documents", "metadatas"],
        )
        if len(documents) >= 1:
            document = documents[0]
            document_metadata = document["metadata"]

        if not document:
            raise Exception("No record could be found for that memory")

        # Fetch file(s)
        new_file_name = embedding.create_parsed_filename(collection_name, document_id)
        tmp_folder = TMP_DOCUMENT_PATH
        tmp_file_path = os.path.join(TMP_DOCUMENT_PATH, new_file_name)
        if url_path:
            # Download the file and save to disk
            print(f"[homebrew api] Downloading file to {tmp_file_path} ...")
            await get_file_from_url(url_path, tmp_file_path)
        elif file_path:
            # Copy file from provided location to /tmp dir, only if paths differ
            print(f"[homebrew api] Loading local file from disk {file_path} ...")
            if file_path != tmp_file_path:
                if not os.path.exists(tmp_folder):
                    os.makedirs(tmp_folder)
                shutil.copy(file_path, tmp_file_path)
            print("[homebrew api] File to be copied already in /tmp dir")
        else:
            raise Exception("Please supply a local path or url to a file")

        # Compare checksums
        updated_document_metadata = {}
        new_file_hash = embedding.create_checksum(tmp_file_path)
        stored_file_hash = document_metadata["checksum"]
        if new_file_hash != stored_file_hash:
            # Pass provided metadata or stored
            updated_document_metadata = metadata or document_metadata
            description = updated_document_metadata["description"]
            # Validate tags
            updated_tags = parse_valid_tags(updated_document_metadata["tags"])
            if updated_tags == None:
                raise Exception("Invalid value for 'tags' input.")
            # Process input documents
            processed_file = embedding.pre_process_documents(
                document_id=document_id,
                document_name=document_name,
                collection_name=collection_name,
                description=description,
                tags=updated_tags,
                input_file_path=tmp_file_path,
                output_folder_path=PARSED_DOCUMENT_PATH,
            )
            # Create text embeddings
            if app.state.llm == None:
                app.state.llm = text_llama_index.load_text_model(
                    app.state.path_to_model
                )
            form = {
                "collection_name": collection_name,
                "document_name": document_name,
                "document_id": document_id,
                "description": description,
                "tags": updated_tags,
                "is_update": True,
            }
            background_tasks.add_task(
                embedding.create_embedding,
                processed_file,
                app.state.storage_directory,
                form,
                app.state.llm,
                db,
            )
        else:
            # Delete tmp files if exist
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            # If same input file, abort
            raise Exception("Input file has not changed.")

        return {
            "success": True,
            "message": f"Updated memories [{document_name}]",
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": f"{e}",
        }


class DeleteDocumentsRequest(BaseModel):
    collection_id: str
    document_ids: List[str]


class DeleteDocumentsResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Removed 1 document(s)",
                    "success": True,
                }
            ]
        }
    }


# Delete a document by id
@app.post("/v1/memory/deleteDocuments")
def delete_documents(params: DeleteDocumentsRequest) -> DeleteDocumentsResponse:
    try:
        collection_id = params.collection_id
        document_ids = params.document_ids
        num_documents = len(document_ids)
        document = None
        db = get_vectordb_client()
        collection = db.get_collection(collection_id)
        sources: List[str] = json.loads(collection.metadata["sources"])
        source_file_path = ""
        documents = embedding.get_document(
            collection_name=collection_id,
            document_ids=document_ids,
            db=db,
            include=["metadatas"],
        )
        if app.state.llm == None:
            app.state.llm = text_llama_index.load_text_model(app.state.path_to_model)
        # Delete all files and references associated with embedded docs
        for document in documents:
            document_metadata = document["metadata"]
            source_file_path = document_metadata["filePath"]
            document_id = document_metadata["id"]
            # Remove file from disk
            print(f"[homebrew api] Remove file {document_id} from {source_file_path}")
            if os.path.exists(source_file_path):
                os.remove(source_file_path)
            # Remove source reference from collection array
            sources.remove(document_id)
            # Update collection
            sources_json = json.dumps(sources)
            collection.metadata["sources"] = sources_json
            collection.modify(metadata=collection.metadata)
            # Delete embeddings from llama-index @TODO Verify this works
            index = embedding.load_embedding(app.state.llm, db, collection_id)
            index.delete(document_id)
        # Delete the embeddings from collection
        collection.delete(ids=document_ids)

        return {
            "success": True,
            "message": f"Removed {num_documents} document(s): {document_ids}",
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": e,
        }


class DeleteCollectionRequest(BaseModel):
    collection_id: str


class DeleteCollectionResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Removed collection",
                    "success": True,
                }
            ]
        }
    }


# Delete a collection by id
@app.get("/v1/memory/deleteCollection")
def delete_collection(
    params: DeleteCollectionRequest = Depends(),
) -> DeleteCollectionResponse:
    try:
        collection_id = params.collection_id
        db = get_vectordb_client()
        collection = db.get_collection(collection_id)
        sources: List[str] = json.loads(collection.metadata["sources"])
        include = ["documents", "metadatas"]
        # Remove all associated source files
        documents = embedding.get_document(
            collection_name=collection_id,
            document_ids=sources,
            db=get_vectordb_client(),
            include=include,
        )
        for document in documents:
            document_metadata = document["metadata"]
            filePath = document_metadata["filePath"]
            os.remove(filePath)
        # Remove the collection
        db.delete_collection(name=collection_id)
        # Remove persisted vector index from disk
        delete_vector_store(collection_id)

        return {
            "success": True,
            "message": f"Removed collection [{collection_id}]",
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": e,
        }


class WipeMemoriesResponse(BaseModel):
    success: bool
    message: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Removed all memories",
                    "success": True,
                }
            ]
        }
    }


# Completely wipe database
@app.get("/v1/memory/wipe")
def wipe_all_memories() -> WipeMemoriesResponse:
    try:
        db = get_vectordb_client()
        # Delete all db values
        db.reset()
        # Delete all parsed documents/files in /memories
        if os.path.exists(TMP_DOCUMENT_PATH):
            files = glob.glob(f"{TMP_DOCUMENT_PATH}/*")
            for f in files:
                os.remove(f)  # del files
            os.rmdir(TMP_DOCUMENT_PATH)  # del folder
        if os.path.exists(PARSED_DOCUMENT_PATH):
            files = glob.glob(f"{PARSED_DOCUMENT_PATH}/*.md")
            for f in files:
                os.remove(f)  # del all .md files
            os.rmdir(PARSED_DOCUMENT_PATH)  # del folder
        # Remove persisted vector storage folder
        if os.path.exists(VECTOR_STORAGE_PATH):
            folders = glob.glob(f"{VECTOR_STORAGE_PATH}/*")
            for dir in folders:
                if not "chroma." in dir:
                    files = glob.glob(f"{dir}/*")
                    for f in files:
                        os.remove(f)  # del files
            os.rmdir(dir)  # del folder

        return {
            "success": True,
            "message": "Successfully wiped all memories from Ai",
        }
    except Exception as e:
        print(f"[homebrew api] Error: {e}")
        return {
            "success": False,
            "message": e,
        }


# Methods...


def delete_vector_store(target_file_path: str):
    path_to_delete = os.path.join(VECTOR_STORAGE_PATH, target_file_path)
    if os.path.exists(path_to_delete):
        files = glob.glob(f"{path_to_delete}/*")
        for f in files:
            os.remove(f)  # del files
        os.rmdir(path_to_delete)  # del folder


# Verify the string contains only lowercase letters, numbers, and a select special chars and whitespace
# In-validate by checking for "None" return value
def parse_valid_tags(tags: str):
    try:
        # Check for correct type of input
        if not isinstance(tags, str):
            raise Exception("'Tags' must be a string")
        # We dont care about empty string for optional input
        if not len(tags):
            return tags
        # Allow only lowercase chars, numbers and certain special chars and whitespaces
        m = re.compile(r"^[a-z0-9$*-]+( [a-z0-9$*-]+)*$")
        if not m.match(tags):
            raise Exception("'Tags' input value has invalid chars.")
        # Remove any whitespace/hyphens from start/end
        result = tags.strip()
        result = tags.strip("-")
        # Remove invalid single words
        array_values = result.split(" ")
        result_array = []
        for word in array_values:
            # Words cannot have dashes at start/end
            p_word = word.strip("-")
            # Single char words not allowed
            if len(word) > 1:
                result_array.append(p_word)
        result = " ".join(result_array)
        # Return a sanitized string
        return result
    except Exception as e:
        print(f"[homebrew api] {e}")
        return None


# Determine if the input string is acceptable as an id
def check_valid_id(input: str):
    l = len(input)
    # Cannot be empty
    if not l:
        return False
    # Check for sequences reserved for our parsing scheme
    matches_double_hyphen = re.findall("--", input)
    if matches_double_hyphen:
        print(f"[homebrew api] Found double hyphen in 'id': {input}")
        return False
    # All names must be 3 and 63 characters
    if l > 63 or l < 3:
        return False
    # No hyphens at start/end
    if input[0] == "-" or input[l - 1] == "-":
        print("[homebrew api] Found hyphens at start/end in [id]")
        return False
    # No whitespace allowed
    matches_whitespace = re.findall("\s", input)
    if matches_whitespace:
        print("[homebrew api] Found whitespace in [id]")
        return False
    # Check special chars. All chars must be lowercase. Dashes acceptable.
    m = re.compile(r"[a-z0-9-]*$")
    if not m.match(input):
        print("[homebrew api] Found invalid special chars in [id]")
        return False
    # Passes
    return True


async def get_file_from_url(url: str, pathname: str):
    # example url: https://raw.githubusercontent.com/dieharders/ai-text-server/master/README.md
    client: httpx.Client = app.requests_client
    CHUNK_SIZE = 1024 * 1024  # 1mb
    TOO_LONG = 751619276  # about 700mb limit in "bytes"
    headers = {
        "Content-Type": "application/octet-stream",
    }
    # @TODO Verify stored checksum before downloading
    head_res = client.head(url)
    total_file_size = head_res.headers.get("content-length")
    if int(total_file_size) > TOO_LONG:
        raise Exception("File is too large")
    # Stream binary content
    with client.stream("GET", url, headers=headers) as res:
        res.raise_for_status()
        if res.status_code != httpx.codes.OK:
            raise Exception("Something went wrong fetching file")
        if int(res.headers["Content-Length"]) > TOO_LONG:
            raise Exception("File is too large")
        with open(pathname, "wb") as file:
            # Write data to disk
            for block in res.iter_bytes(chunk_size=CHUNK_SIZE):
                file.write(block)
    return True


# Open a native file explorer at location of given source
def file_explore(path: str):
    # explorer would choke on forward slashes
    path = os.path.normpath(path)

    if os.path.isdir(path):
        subprocess.run([FILEBROWSER_PATH, path])
    elif os.path.isfile(path):
        subprocess.run([FILEBROWSER_PATH, "/select,", path])


def parse_mentions(input_string) -> Tuple[List[str], str]:
    # Pattern match words starting with @ at the beginning of the string
    pattern = r"^@(\w+)"

    # Find the match at the beginning of the string
    matches = re.findall(pattern, input_string)

    # Check if there is a match
    if matches:
        # Remove the matched words from the original string
        base_query = re.sub(pattern, "", input_string)
        print(f"Found mentions starting with @: {matches}")
        return [matches, base_query]
    else:
        return [[], input_string]


def get_vectordb_client():
    if app.state.db_client == None:
        app.state.db_client = embedding.create_db_client(app.state.storage_directory)
    return app.state.db_client


def token_streamer(token_generator):
    # @TODO We may need to do some token parsing here...multi-byte encoding can cut off emoji/khanji chars.
    # result = "" # accumulate a final response to be encoded in utf-8 in entirety
    try:
        for token in token_generator:
            payload = {"event": "PAYLOAD", "data": f"{token}"}
            yield json.dumps(payload)
    except (ValueError, UnicodeEncodeError, Exception) as e:
        msg = f"Error streaming tokens: {e}"
        print(msg)
        raise Exception(msg)


# Belongs in text inference module
def query_memory(query: str, collection_names: List[str]):
    if app.state.llm == None:
        app.state.llm = text_llama_index.load_text_model(app.state.path_to_model)
    # @TODO We can do filtering based on doc/collection name, metadata, etc via LlamaIndex.
    collection_name = collection_names[0]  # Only take the first collection for now
    db = get_vectordb_client()
    indexDB = embedding.load_embedding(app.state.llm, db, collection_name)
    # Stream the response
    token_generator = embedding.query_embedding(query, indexDB)
    return token_generator


def kill_text_inference():
    if hasattr(app, "text_inference_process"):
        if app.text_inference_process.poll() != None:
            app.text_inference_process.kill()
            app.text_inference_process = None


def start_homebrew_server():
    try:
        print("[homebrew api] Starting API server...")
        # Start the ASGI server
        uvicorn.run(app, host="0.0.0.0", port=app.PORT_HOMEBREW_API, log_level="info")
        return True
    except:
        print("[homebrew api] Failed to start API server")
        return False


if __name__ == "__main__":
    # Determine path to file based on prod or dev
    current_directory = os.getcwd()
    substrings = current_directory.split("\\")
    last_substring = substrings[-1]
    if last_substring == "backends":
        path = "../shared/constants.json"
    else:
        path = "./shared/constants.json"
    # Open and read the JSON constants file
    with open(path, "r") as json_file:
        data = json.load(json_file)
        app.PORT_HOMEBREW_API = data["PORT_HOMEBREW_API"]
        app.PORT_TEXT_INFERENCE = data["PORT_TEXT_INFERENCE"]
    # Starts the homebrew API server
    start_homebrew_server()
