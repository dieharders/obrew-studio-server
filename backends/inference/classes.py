from enum import Enum
from types import NoneType
from typing import Any, List, Optional
from pydantic import BaseModel
from core.classes import KnowledgeSettings
from tools.classes import DEFAULT_TOOL_SCHEMA_TYPE


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Chat message."""

    role: str = MessageRole.USER.value
    content: Optional[str] = ""


class ChatHistory(BaseModel):
    messages: List[ChatMessage]


class RetrievalTypes(str, Enum):
    BASE = "base"
    AUGMENTED = "augmented"
    AGENT = "agent"


class CHAT_MODES(str, Enum):
    INSTRUCT = "instruct"
    CHAT = "chat"
    COLLAB = "collab"


class TOOL_RESPONSE_MODES(str, Enum):
    ANSWER = "answer"
    RESULT = "result"


# class ACTIVE_ROLES(str, Enum):
#     WORKER = "worker"
#     AGENT = "agent"


class TOOL_USE_MODES(str, Enum):
    NATIVE = "native"
    UNIVERSAL = "universal"


DEFAULT_TEMPERATURE = 0.8
DEFAULT_CHAT_MODE = CHAT_MODES.INSTRUCT.value
DEFAULT_TOOL_RESPONSE_MODE = TOOL_RESPONSE_MODES.ANSWER.value
DEFAULT_TOOL_USE_MODE = TOOL_USE_MODES.UNIVERSAL.value
DEFAULT_MAX_TOKENS = (
    -2
)  # until end of context window @TODO may not be good for chat mode?
DEFAULT_SEED = 1337
DEFAULT_CONTEXT_WINDOW = 0
DEFAULT_MIN_CONTEXT_WINDOW = 2000

# More templates found here: https://github.com/run-llama/llama_index/blob/main/llama_index/prompts/default_prompts.py
DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
"""


class AgentOutput(BaseModel):
    text: str  # current tokenized output or entire output
    raw: Optional[Any] = None  # tokens, including template tokens
    logging: Optional[List[str]] = None
    metrics: Optional[dict] = None


class SSEResponse(BaseModel):
    event: str
    data: Optional[AgentOutput] = None


class RagTemplateData(BaseModel):
    id: str
    name: str
    text: str
    type: Optional[str] = None


class LoadTextInferenceCall(BaseModel):
    stream: Optional[bool] = True
    stop: Optional[str] = ""
    echo: Optional[bool] = False
    model: Optional[str] = "local"
    mirostat_tau: Optional[float] = 5.0
    tfs_z: Optional[float] = 1.0
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    temperature: Optional[float] = DEFAULT_TEMPERATURE
    grammar: Optional[dict] = None
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS


class LoadTextInferenceInit(BaseModel):
    n_gpu_layers: Optional[int] = 0  # 32 for our purposes
    use_mmap: Optional[bool] = True
    use_mlock: Optional[bool] = True
    # optimal choice depends on balancing memory constraints and performance requirements
    # Allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
    cache_type_k: Optional[str] = "f16"
    cache_type_v: Optional[str] = "f16"
    seed: Optional[int] = DEFAULT_SEED
    n_ctx: Optional[int] = DEFAULT_CONTEXT_WINDOW  # load from model
    n_batch: Optional[int] = 2048
    n_threads: Optional[int] = -1  # -1 all available
    offload_kqv: Optional[bool] = True
    verbose: Optional[bool] = False


# Load in the ai model to be used for inference
class LoadInferenceRequest(BaseModel):
    modelId: str
    modelPath: Optional[str] = None  # If not provided, looked up from modelId
    raw_input: Optional[bool] = False  # user can send manually formatted messages
    responseMode: Optional[str] = DEFAULT_CHAT_MODE
    toolUseMode: Optional[str] = DEFAULT_TOOL_USE_MODE
    toolSchemaType: Optional[str] = DEFAULT_TOOL_SCHEMA_TYPE
    init: LoadTextInferenceInit
    call: LoadTextInferenceCall
    messages: Optional[List[ChatMessage]] = None


class LoadedTextModelResData(BaseModel):
    modelId: str
    responseMode: str = None
    modelSettings: LoadTextInferenceInit
    generateSettings: LoadTextInferenceCall


class LoadedTextModelResponse(BaseModel):
    success: bool
    message: str
    data: LoadedTextModelResData


class LoadInferenceResponse(BaseModel):
    message: str
    success: bool
    data: NoneType

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "AI model [llama-2-13b-chat-ggml] loaded.",
                    "success": True,
                    "data": None,
                }
            ]
        }
    }


# @TODO This should extend LoadTextInferenceCall (or vice versa)
class InferenceRequest(BaseModel):
    # __init__ args
    n_ctx: Optional[int] = DEFAULT_CONTEXT_WINDOW
    seed: Optional[int] = DEFAULT_SEED
    # homebrew server specific args
    tools: Optional[List[str]] = None
    memory: Optional[KnowledgeSettings] = None
    responseMode: Optional[str] = DEFAULT_CHAT_MODE
    toolResponseMode: Optional[str] = DEFAULT_TOOL_RESPONSE_MODE
    systemMessage: Optional[str] = None
    messageFormat: Optional[str] = None
    promptTemplate: Optional[str] = None
    # __call__ args
    prompt: str
    messages: Optional[List[ChatMessage]] = None
    stream: Optional[bool] = True
    temperature: Optional[float] = 0.0  # precise
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    stop: Optional[str] = None  # Stops generation when string encountered
    echo: Optional[bool] = False
    model: Optional[str] = (
        "local"  # The name to use for the model in the completion object
    )
    grammar: Optional[dict] = None  # A grammar to use for constrained sampling
    mirostat_tau: Optional[float] = (
        5.0  # A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    )
    tfs_z: Optional[float] = (
        1.0  # Tail Free Sampling - https://www.trentonbricken.com/Tail-Free-Sampling/
    )
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1
    presence_penalty: Optional[float] = (
        0.0  # The penalty to apply to tokens based on their presence in the prompt
    )
    frequency_penalty: Optional[float] = (
        0.0  # The penalty to apply to tokens based on their frequency in the prompt
    )
    similarity_top_k: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Why does mass conservation break down?",
                    "memory": {"ids": ["science"]},
                    "tools": ["calculator"],
                    "responseMode": DEFAULT_CHAT_MODE,
                    "systemMessage": "You are a helpful Ai assistant.",
                    "messageFormat": "<system> {{system_message}}\n<user> {{user_message}}",
                    "promptTemplate": "Answer this question: {{user_prompt}}",
                    "messages": [
                        {"role": "user", "content": "What is meaning of life?"}
                    ],
                    # Settings
                    "stream": True,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                    "n_ctx": 2000,
                    "stop": "[DONE]",
                    "echo": False,
                    "model": "llama2",
                    "grammar": None,
                    "mirostat_tau": 5.0,
                    "tfs_z": 1.0,
                    "top_k": 40,
                    "top_p": 0.95,
                    "min_p": 0.05,
                    "seed": 1337,
                    "repeat_penalty": 1.1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "similarity_top_k": 1,
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


# Vision/Multi-modal classes
class VisionInferenceRequest(BaseModel):
    """Request for vision inference with image input."""

    prompt: str
    images: List[str]  # Base64 encoded images or file paths
    image_type: Optional[str] = "base64"  # "base64" or "path"
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    systemMessage: Optional[str] = None
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.95
    min_p: Optional[float] = 0.05
    repeat_penalty: Optional[float] = 1.1

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "What objects are in this image?",
                    "images": ["<base64_encoded_image>"],
                    "image_type": "base64",
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 1024,
                }
            ]
        }
    }


class LoadVisionInferenceRequest(BaseModel):
    """Load a vision model with mmproj file."""

    modelId: str
    modelPath: Optional[str] = None  # If not provided, looked up from modelId
    mmprojPath: Optional[str] = None  # If not provided, looked up from modelId
    init: LoadTextInferenceInit
    call: LoadTextInferenceCall


# Image Embedding classes
class VisionEmbedRequest(BaseModel):
    """Request to create embedding for an image and store in ChromaDB."""

    image_path: Optional[str] = None  # Path to image file
    image_base64: Optional[str] = None  # Base64 encoded image
    image_type: Optional[str] = "path"  # "path" or "base64"
    collection_name: Optional[str] = (
        None  # ChromaDB collection (auto-create from filename if not provided)
    )
    description: Optional[str] = None
    metadata: Optional[dict] = None
    repo_id: Optional[str] = None  # Model repo ID from frontend settings

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "image_path": "/path/to/image.jpg",
                    "image_type": "path",
                    "collection_name": "my_images",
                    "description": "Description of the image.",
                    "repo_id": "owner/model-name",
                }
            ]
        }
    }


class VisionEmbedLoadRequest(BaseModel):
    """Request to load a multimodal embedding model."""

    model_path: str  # Path to GGUF model file
    mmproj_path: str  # Path to mmproj file
    model_name: Optional[str] = None  # Friendly name
    model_id: Optional[str] = None  # Model identifier
    port: Optional[int] = 8081  # Port for embedding server
    n_gpu_layers: Optional[int] = 99  # GPU layers to offload
    n_threads: Optional[int] = 4  # CPU threads
    n_ctx: Optional[int] = 2048  # Context window

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_path": "/path/to/model.gguf",
                    "mmproj_path": "/path/to/mmproj.gguf",
                    "model_name": "jina-embeddings-v4",
                    "n_gpu_layers": 99,
                }
            ]
        }
    }


class DownloadVisionEmbedModelRequest(BaseModel):
    """Request to download a vision embedding model from HuggingFace."""

    repo_id: str  # HuggingFace repo ID
    filename: str  # Model GGUF filename
    mmproj_filename: str  # mmproj GGUF filename

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "repo_id": "mradermacher/GME-VARCO-VISION-Embedding-GGUF",
                    "filename": "GME-VARCO-VISION-Embedding.Q4_K_M.gguf",
                    "mmproj_filename": "GME-VARCO-VISION-Embedding.mmproj-Q8_0.gguf",
                }
            ]
        }
    }


class DeleteVisionEmbedModelRequest(BaseModel):
    """Request to delete a vision embedding model."""

    repo_id: str  # HuggingFace repo ID


class VisionEmbedQueryRequest(BaseModel):
    """Request to query/search an image collection using text."""

    query: str  # Text query to search for
    collection_name: str  # ChromaDB collection to search
    top_k: Optional[int] = 5  # Number of results to return
    include_embeddings: Optional[bool] = False  # Include embedding vectors in response

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "a red sports car on a mountain road",
                    "collection_name": "my_images",
                    "top_k": 5,
                    "include_embeddings": False,
                }
            ]
        }
    }


# Agentic File Search classes
class FileSearchRequest(BaseModel):
    """Request for agentic file search."""

    query: str  # The search query
    directory: str  # Directory to search in
    allowed_directories: List[str]  # Whitelist of directories the agent can access
    file_patterns: Optional[List[str]] = None  # File extensions to filter (e.g., [".pdf", ".docx"])
    max_files_preview: Optional[int] = 10  # Max files to preview
    max_files_parse: Optional[int] = 3  # Max files to fully parse
    cache_results: Optional[bool] = False  # Cache parsed docs in ChromaDB
    collection_name: Optional[str] = None  # Collection for caching (required if cache_results=True)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find all documents about quarterly sales reports",
                    "directory": "/documents/reports",
                    "allowed_directories": ["/documents/reports", "/documents/archives"],
                    "file_patterns": [".pdf", ".docx"],
                    "max_files_preview": 10,
                    "max_files_parse": 3,
                    "cache_results": False,
                }
            ]
        }
    }


class FileSearchResponse(BaseModel):
    """Response from agentic file search."""

    success: bool
    message: str
    data: Optional[dict] = None  # Contains: answer, sources, tool_logs, etc.


# True Agentic Search classes (LLM chooses tools)
class AgenticSearchRequest(BaseModel):
    """Request for true agentic file search where LLM decides which tools to use."""

    query: str  # The search query
    directory: str  # Starting directory to search in
    allowed_directories: List[str]  # Whitelist of directories the agent can access
    max_iterations: Optional[int] = 10  # Maximum tool calls before stopping
    file_patterns: Optional[List[str]] = None  # Hints about file types to focus on
    cache_results: Optional[bool] = False  # Cache results in ChromaDB
    collection_name: Optional[str] = None  # Collection for caching

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Find the function that handles user authentication",
                    "directory": "/projects/myapp/src",
                    "allowed_directories": ["/projects/myapp"],
                    "max_iterations": 10,
                    "file_patterns": [".py", ".js"],
                }
            ]
        }
    }


class AgenticSearchResponse(BaseModel):
    """Response from true agentic file search."""

    success: bool
    message: str
    data: Optional[dict] = None  # Contains: answer, sources, tool_logs, iterations
