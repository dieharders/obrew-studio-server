import httpx
from types import NoneType
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union
from enum import Enum
from chromadb import Collection
from chromadb.api import ClientAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from inference.llama_cpp import LLAMA_CPP
from fastapi import FastAPI


class AppState(dict):
    requests_client: httpx.Client
    PORT_HOMEBREW_API: int
    db_client: ClientAPI
    llm: LLAMA_CPP | None
    embed_model: HuggingFaceEmbedding | str


class FastAPIApp(FastAPI):
    state: AppState


class FILE_LOADER_SOLUTIONS(Enum):
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
    mode: str = None


class PerformanceSettings(BaseModel):
    n_gpu_layers: int = None
    use_mlock: bool = None
    seed: int = None
    n_ctx: int = None
    n_batch: int = None
    n_threads: int = None
    offload_kqv: bool = None
    chat_format: str = None
    cache_type_k: str = None
    cache_type_v: str = None


class ToolsSettings(BaseModel):
    assigned: List[str | None] = None


class SystemSettings(BaseModel):
    systemMessage: str = None
    systemMessageName: str = None


class ModelSettings(BaseModel):
    id: str = None  # @TODO change to modelId
    filename: str = None
    botName: Optional[str] = None


class PromptSettings(BaseModel):
    promptTemplate: dict = None
    ragTemplate: dict = None
    ragMode: dict = None


class KnowledgeSettings(BaseModel):
    type: str = None
    index: List[str | None] = None


class ResponseSettings(BaseModel):
    temperature: float = None
    max_tokens: int = None
    top_p: float = None
    echo: bool = None
    stop: List[str] = []
    repeat_penalty: float = None
    top_k: int = None
    stream: bool = None


class ToolDefinition(BaseModel):
    name: str
    path: str
    arguments: Optional[dict | None] = None
    example_arguments: Optional[dict | None] = None
    id: Optional[str] = None
    description: Optional[str] = ""


class ToolSaveRequest(BaseModel):
    name: str
    path: str
    id: Optional[str] = None  # pass string to edit tool, leave blank to add new tool

    @field_validator("id")
    @classmethod
    def prevent_none(cls, v):
        assert v is not None, "id may not be None"
        assert v != "", "id may not be empty"
        return v


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
    knowledge: KnowledgeSettings = None
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
                    "savePath": "C:\\Project Files\\brain-dump-ai\\models\\llama-2-13b-chat.ggmlv3.q2_K.bin",
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
