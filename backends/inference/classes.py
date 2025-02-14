from enum import Enum
from types import NoneType
from typing import List, Optional
from pydantic import BaseModel


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


class ChatMessage(BaseModel):
    """Chat message."""

    role: MessageRole = MessageRole.USER
    content: Optional[str] = ""


class ChatHistory(BaseModel):
    messages: List[ChatMessage]


class RetrievalTypes(Enum):
    BASE = "base"
    AUGMENTED = "augmented"
    AGENT = "agent"


class CHAT_MODES(Enum):
    INSTRUCT = "instruct"
    CHAT = "chat"
    COLLAB = "collab"


DEFAULT_TEMPERATURE = 0.2
DEFAULT_CHAT_MODE = CHAT_MODES.INSTRUCT
DEFAULT_MAX_TOKENS = -2  # until end of context window
DEFAULT_SEED = 1337
DEFAULT_CONTEXT_WINDOW = 0
DEFAULT_MIN_CONTEXT_WINDOW = 2000

# More templates found here: https://github.com/run-llama/llama_index/blob/main/llama_index/prompts/default_prompts.py
DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
"""


class RagTemplateData(BaseModel):
    id: str
    name: str
    text: str
    type: Optional[str] = None


class LoadTextInferenceCall(BaseModel):
    stream: Optional[bool] = True
    stop: Optional[List[str]] = []
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
    use_mlock: Optional[bool] = False
    # optimal choice depends on balancing memory constraints and performance requirements
    # Allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
    cache_type_k: Optional[str] = "f16"
    seed: Optional[int] = DEFAULT_SEED
    n_ctx: Optional[int] = DEFAULT_CONTEXT_WINDOW  # load from model
    n_batch: Optional[int] = 2048
    n_threads: Optional[int] = None
    offload_kqv: Optional[bool] = True
    verbose: Optional[bool] = False


# Load in the ai model to be used for inference
class LoadInferenceRequest(BaseModel):
    modelPath: str
    modelId: str
    raw: Optional[bool] = False  # user can send manually formatted messages
    mode: Optional[str] = DEFAULT_CHAT_MODE
    init: LoadTextInferenceInit
    call: LoadTextInferenceCall
    messages: Optional[List[ChatMessage]] = None


class LoadedTextModelResData(BaseModel):
    modelId: str
    mode: str = None
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


class InferenceRequest(BaseModel):
    # __init__ args
    n_ctx: Optional[int] = DEFAULT_CONTEXT_WINDOW
    seed: Optional[int] = DEFAULT_SEED
    # homebrew server specific args
    collectionNames: Optional[List[str]] = []
    tools: Optional[List[str]] = []
    retrievalType: Optional[RetrievalTypes | None] = None
    mode: Optional[str] = DEFAULT_CHAT_MODE
    systemMessage: Optional[str] = None
    messageFormat: Optional[str] = None
    promptTemplate: Optional[str] = None
    ragPromptTemplate: Optional[RagTemplateData] = None
    # __call__ args
    prompt: str
    messages: Optional[List[str]] = []
    stream: Optional[bool] = True
    # suffix: Optional[str] = ""
    temperature: Optional[float] = 0.0  # precise
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    # A list of strings to stop generation when encountered
    stop: Optional[List[str]] = []
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
    response_mode: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Why does mass conservation break down?",
                    "collectionNames": ["science"],
                    "tools": ["calculator"],
                    "mode": DEFAULT_CHAT_MODE,
                    "systemMessage": "You are a helpful Ai assistant.",
                    "messageFormat": "<system> {system_message}\n<user> {prompt}",
                    "promptTemplate": "Answer this question: {query_str}",
                    "ragPromptTemplate": {
                        "id": "summary",
                        "name": "Summary",
                        "text": "This is a template: {query_str}",
                    },
                    "messages": [
                        {"role": "user", "content": "What is meaning of life?"}
                    ],
                    # Settings
                    "stream": True,
                    "temperature": 0.2,
                    "max_tokens": 1024,
                    "n_ctx": 2000,
                    "stop": ["###", "[DONE]"],
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
                    "response_mode": "compact",
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
