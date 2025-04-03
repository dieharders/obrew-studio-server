import json
import asyncio
import httpx
from enum import Enum
from types import NoneType
from fastapi import FastAPI
from typing import List, Optional, Union, Type
from pydantic import BaseModel
from chromadb import Collection
from chromadb.api import ClientAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from inference.llama_cpp import LLAMA_CPP
from collections.abc import Callable


class ApiServerClass(dict):
    remote_url: str
    SERVER_HOST: str
    SERVER_PORT: int
    SSL_ENABLED: bool
    XHR_PROTOCOL: str
    is_prod: bool
    is_dev: bool
    is_debug: bool
    selected_webui_url: str
    on_startup_callback: Callable
    package_json: json
    api_version: str
    is_prod: bool
    SSL_KEY: str
    SSL_CERT: str
    CUSTOM_ORIGINS_ENV: str
    CUSTOM_ORIGINS = List[str]
    origins: List[str]
    app: FastAPI


class AppState(dict):
    requests_client: Type[httpx.Client]
    request_queue: Type[asyncio.Queue] | None
    db_client: Type[ClientAPI] | None
    api: Type[ApiServerClass] | None
    llm: LLAMA_CPP | None
    embed_model: Type[HuggingFaceEmbedding] | str | None


class FastAPIApp(FastAPI):
    state: AppState


class FILE_LOADER_SOLUTIONS(str, Enum):
    LLAMA_PARSE = "llama_parse"
    READER = "reader_api"
    DEFAULT = "default"


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
                    "data": {
                        "docs": "https://localhost:8008/docs",
                        "version": "0.2.0",
                    },
                }
            ]
        }
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


class GetModelInfoRequest(BaseModel):
    repoId: str


class DownloadTextModelRequest(BaseModel):
    repo_id: str
    filename: str


class DeleteTextModelRequest(BaseModel):
    filename: str
    repoId: str


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


class AddCollectionRequest(BaseModel):
    collectionName: str
    description: Optional[str] = ""
    tags: Optional[str] = ""
    icon: Optional[str] = ""


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


class EmbedDocumentRequest(BaseModel):
    collectionName: str
    documentName: str
    description: Optional[str] = ""
    tags: Optional[str] = ""
    # optional for add, required for update
    documentId: Optional[str] = ""
    # need oneof urlPath/fileName/textInput
    urlPath: Optional[str] = ""
    # need oneof urlPath/fileName/textInput
    filePath: Optional[str] = ""
    # need oneof urlPath/fileName/textInput
    textInput: Optional[str] = ""
    # Chunking settings
    chunkSize: Optional[int] = None
    chunkOverlap: Optional[int] = None
    chunkStrategy: Optional[str] = None
    # Parsing method
    parsingMethod: Optional[str] = None


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
                                "tags": "html5 react",
                                "sources": [
                                    {
                                        "id": "document-id",
                                        "name": "document-name",
                                        "description": "Some description.",
                                        "chunkIds": ["id1", "id2"],
                                    }
                                ],
                            },
                        }
                    ],
                }
            ]
        }
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
    data: Collection

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "Returned collection",
                    "success": True,
                    "data": {},
                }
            ]
        }
    }


class GetDocumentChunksRequest(BaseModel):
    collectionId: str
    documentId: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collectionId": "coll-id",
                    "documentId": "doc-id",
                }
            ]
        }
    }


class GetAllDocumentsRequest(BaseModel):
    collection_id: str
    include: Optional[List[str]] = None


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


class SourceMetadata(dict):
    id: str
    checksum: str
    fileType: str  # type of the source (ingested) file
    filePath: str  # path to parsed file
    fileName: str  # name of parsed file
    fileSize: int  # bytes
    name: str  # document name
    description: str
    tags: str
    createdAt: str
    modifiedLast: str
    chunkIds = List[str]


class FileExploreRequest(BaseModel):
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


# @TODO Extend from LoadTextInferenceInit
# Not used yet
class AppSettingsInitData(BaseModel):
    preset: Optional[str] = None
    n_ctx: Optional[int] = None
    seed: Optional[int] = None
    n_threads: Optional[int] = None
    n_batch: Optional[int] = None
    offload_kqv: Optional[bool] = None
    n_gpu_layers: Optional[int] = None
    cache_type_k: Optional[str] = None
    cache_type_v: Optional[str] = None
    use_mlock: Optional[bool] = None
    use_mmap: Optional[bool] = None
    verbose: Optional[bool] = None


class AttentionSettings(BaseModel):
    response_mode: str = None
    tool_response_mode: str = None
    tool_use_mode: str = None


class PerformanceSettings(BaseModel):
    n_gpu_layers: int = None
    use_mlock: bool = None
    seed: int = None
    n_ctx: int = None
    n_batch: int = None
    n_threads: int = None
    offload_kqv: bool = None
    cache_type_k: str = None
    cache_type_v: str = None


class ToolsSettings(BaseModel):
    assigned: List[str | None] = None


class SystemSettings(BaseModel):
    systemMessage: Optional[str] = None
    systemMessageName: Optional[str] = None


class ModelSettings(BaseModel):
    id: str = None  # @TODO change to modelId
    filename: str = None
    botName: Optional[str] = None


class PromptSettings(BaseModel):
    promptTemplate: dict = None


class KnowledgeSettings(BaseModel):
    type: str = None
    index: List[str | None] = None


# @TODO Extend this from the other dupes
class ResponseSettings(BaseModel):
    temperature: float = None
    max_tokens: int = None
    top_p: float = None
    echo: bool = None
    stop: str = ""
    repeat_penalty: float = None
    top_k: int = None
    stream: bool = None


class ToolFunctionParameter(BaseModel):
    name: str
    title: str
    description: str
    type: str
    placeholder: Optional[str] = None
    input_type: Optional[str] = None
    default_value: Optional[str | int | float | dict | list] = None
    value: Optional[str | int | float | dict | list] = None
    min_value: Optional[int | float | str] = None
    max_value: Optional[int | float | str] = None
    options_source: Optional[str] = None
    options: Optional[List[str]] = None
    items: Optional[dict] = None


class ToolFunctionSchema(BaseModel):
    description: Optional[str] = None
    params: List[ToolFunctionParameter]
    # Tool use
    json_schema: Optional[str] = None
    typescript_schema: Optional[str] = None
    # Schema for params to pass for tool use
    params_schema: Optional[dict] = None
    # All required params with example values
    params_example: Optional[dict] = None
    # The return type of the output
    output_type: Optional[List[str]] = None


class ToolDefinition(ToolFunctionSchema):
    name: str
    path: str
    id: Optional[str] = None


# class ToolSaveRequest(ToolDefinition):
#     id: Optional[str] = None  # pass string to edit tool, leave blank to add new tool

#     @field_validator("id")
#     @classmethod
#     def prevent_none(cls, v):
#         assert v is not None, "id may not be None"
#         assert v != "", "id may not be empty"
#         return v


class GetToolFunctionSchemaResponse(BaseModel):
    success: bool
    message: str
    data: ToolFunctionSchema | None


class ListToolFunctionsResponse(BaseModel):
    success: bool
    message: str
    data: List[str | None]


class GetToolSettingsResponse(BaseModel):
    success: bool
    message: str
    data: Union[List[ToolDefinition], None]


class EmptyToolSettingsResponse(BaseModel):
    success: bool
    message: str
    data: NoneType

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Returned 1 tool.",
                    "data": None,
                }
            ]
        }
    }


class BotSettings(BaseModel):
    tools: ToolsSettings = None
    attention: AttentionSettings = None
    performance: PerformanceSettings = None
    system: SystemSettings = None
    model: ModelSettings = None
    prompt: PromptSettings = None
    response: ResponseSettings = None


class BotSettingsResponse(BaseModel):
    success: bool
    message: str
    data: List[BotSettings] = None


class SaveSettingsRequest(BaseModel):
    data: dict


class GenericEmptyResponse(BaseModel):
    success: bool
    message: str
    data: NoneType

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "This is a message",
                    "success": True,
                    "data": None,
                }
            ]
        }
    }


class InstalledTextModelMetadata(BaseModel):
    repoId: Optional[str] = None
    savePath: Optional[str | dict] = None
    numTimesRun: Optional[int] = None
    isFavorited: Optional[bool] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "llama2-13b",
                    "savePath": "C:\\project\\models\\llama-2-13b-chat.ggmlv3.q2_K.bin",
                    "numTimesRun": 0,
                    "isFavorited": False,
                }
            ]
        }
    }


class InstalledTextModel(BaseModel):
    current_download_path: str
    installed_text_models: List[InstalledTextModelMetadata]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "current_download_path": "C:\\Users\\user\\Downloads\\llama-2-13b-chat.Q4_K_M.gguf",
                    "installed_text_models": [
                        {
                            "id": "llama-2-13b-chat",
                            "savePath": {
                                "llama-2-13b-chat-Q5_1": "C:\\Users\\user\\Downloads\\llama-2-13b-chat.Q4_K_M.gguf"
                            },
                            "numTimesRun": 0,
                            "isFavorited": False,
                        }
                    ],
                }
            ]
        }
    }


class TextModelInstallMetadataResponse(BaseModel):
    success: bool
    message: str
    data: List[InstalledTextModelMetadata]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Success",
                    "data": [
                        {
                            "id": "llama-2-13b-chat",
                            "savePath": {
                                "llama-2-13b-chat-Q5_1": "C:\\Users\\user\\Downloads\\llama-2-13b-chat.Q4_K_M.gguf"
                            },
                            "numTimesRun": 0,
                            "isFavorited": False,
                        }
                    ],
                }
            ]
        }
    }


class InstalledTextModelResponse(BaseModel):
    success: bool
    message: str
    data: InstalledTextModelMetadata

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Success",
                    "data": {
                        "id": "llama-2-13b-chat",
                        "savePath": {
                            "llama-2-13b-chat-Q5_1": "C:\\Users\\user\\Downloads\\llama-2-13b-chat.Q4_K_M.gguf"
                        },
                        "numTimesRun": 0,
                        "isFavorited": False,
                    },
                }
            ]
        }
    }


class ContextRetrievalOptions(BaseModel):
    response_mode: Optional[str] = None
    similarity_top_k: Optional[int] = None
