import time
import os
import subprocess
import requests
import atexit
from collections.abc import Callable
from typing import List, Optional, Sequence
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from core import common, classes

# More templates found here: https://github.com/run-llama/llama_index/blob/main/llama_index/prompts/default_prompts.py
DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
"""


# Run a llama.cpp server binary wrapped in OpenAI api compatible-ish class
class Llama_CPP_SERVER:
    def __init__(
        self,
        model_url: str,  # url to download from
        model_path: str,  # file path
        model_name: str,  # friendly name
        mode: str,
        messages_to_prompt: Optional[Callable] = None,  # @TODO Do something with
        completion_to_prompt: Optional[Callable] = None,  # @TODO Do something with
        host="127.0.0.1",
        port=8090,
        verbose=False,
        debug=False,
        model_init_kwargs: dict = None,
        generate_kwargs: dict = None,
        priority: int = 0,  # set process priority (0=default, 1, 2, 3)
    ):
        self.process = None
        self.mode = mode
        self.host = host
        self.port = port
        self.model_url = model_url
        self.model_name = model_name or "chatbot"  # human friendly name for display
        self.verbose = verbose
        self.debug = debug
        self.priority = priority
        self.model_path = model_path  # text-models/your-model.gguf
        self.model_init_kwargs = model_init_kwargs
        self._generate_kwargs = generate_kwargs
        BINARY_BASE_PATH = "servers"
        BINARY_FOLDER = "llama.cpp"
        self.BINARY_PATH: str = os.path.join(
            os.getcwd(), BINARY_BASE_PATH, BINARY_FOLDER, "llama-server.exe"
        )

    # Getter
    @property
    def generate_kwargs(self) -> dict:
        return self._generate_kwargs

    # Translate settings from UI to what inference server expects
    @generate_kwargs.setter
    def generate_kwargs(self, settings: dict):
        n_ctx = self.model_init_kwargs.get("--ctx-size") or 0
        if n_ctx <= 0:
            n_ctx = 0
        kwargs = {
            "stream": settings.get("stream"),
            "stop": settings.get("stop"),  # !Never use an empty string like [""]
            "verbose-prompt": settings.get("echo"),
            "model": settings.get("model"),
            "mirostat_tau": settings.get("mirostat_tau"),
            "tfs_z": settings.get("tfs_z"),
            "top_k": settings.get("top_k"),
            "top_p": settings.get("top_p"),
            "min_p": settings.get("min_p"),
            "repeat_penalty": settings.get("repeat_penalty"),
            "presence_penalty": settings.get("presence_penalty"),
            "frequency_penalty": settings.get("frequency_penalty"),
            "temperature": settings.get("temperature"),
            "seed": self.model_init_kwargs.get("--seed"),
            "grammar": settings.get("grammar"),
            "n-predict": common.calc_max_tokens(
                settings.get("max_tokens"), n_ctx, self.mode
            ),
        }
        self._generate_kwargs = kwargs

    # Shutdown server
    def unload(self):
        if self.process:
            print(
                f"{common.PRNT_LLAMA} Shutting down llama.cpp server process.",
                flush=True,
            )
            self.process.kill()

    # Start http server
    def load_model(self):
        """
        Starts the llama.cpp server using subprocess.
        """
        try:
            # Create arguments for starting server
            cmd_args = [
                self.BINARY_PATH,
                "--model",
                self.model_path,
                "--host",
                self.host,
                "--port",
                str(self.port),
                "--prio-batch",
                str(self.priority),
                "--no-webui",
                # "--log-prefix", # enable to set prefix to logs from .env: LLAMA_LOG_PREFIX
                # "--version", # use to see version, but will self-terminate
            ]
            if self.debug:
                cmd_args.append("--log-verbose")
            # Sanitize values
            for key, value in self.model_init_kwargs.items():
                if isinstance(value, bool):
                    if value == True:
                        cmd_args.append(key)
                    else:
                        pass
                else:
                    cmd_args.extend([f"{key}", str(value)])

            # Start server process
            print(
                f"{common.PRNT_LLAMA} Starting llama.cpp server on port {self.port}..."
            )
            process = subprocess.Popen(cmd_args, text=True)

            # Define the health check endpoint
            health_url = f"http://localhost:{self.port}/health"
            headers = {"Authorization": "Bearer no-key"}
            start_time = time.time()
            timeout = 30000  # 30 seconds

            # Wait (health check) for the server to be ready
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url=health_url, headers=headers)
                    if response.status_code == 200:
                        print(
                            f"{common.PRNT_LLAMA} Server is ready to accept requests."
                        )
                        self.process = process
                        # Register a cleanup to occur on python interpreter exit
                        atexit.register(self.unload)
                        return process
                except requests.ConnectionError:
                    pass
                time.sleep(1)

            print(f"{common.PRNT_LLAMA} Terminating server, timed out.")
            process.terminate()
            raise Exception("Server timed out")

        except Exception as e:
            print(f"{common.PRNT_LLAMA} Error starting the server: {e}")
            raise Exception(f"Error starting the server: {e}")

    # Predict the rest of the prompt
    # @TODO do something with message_format
    def complete(
        self, prompt: str, system_message: str, message_format: str, is_chat=False
    ):
        """
        Queries the /completions endpoint of the llama.cpp server.
        """
        chat_endpoint = "/chat" if is_chat else ""
        if self.host == "127.0.0.1":
            host = "http://localhost"
        else:
            host = self.host
        url = f"{host}:{self.port}/v1{chat_endpoint}/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer no-key"}
        payload = {
            "prompt": prompt,  # "prompt" for non-chat v1/completions
            # @TODO Handle "messages": [] for chat mode v1/chat/completions
        }
        if system_message:
            payload["system_message"] = system_message
        payload.update(self.generate_kwargs)

        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"{common.PRNT_LLAMA} Error querying the server: {e}")
            raise Exception(f"Error querying the server: {e}")


# Format the prompt for chat conversations
# @TODO Could also use: from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt
def _messages_to_prompt(
    messages: Sequence[ChatMessage],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
    template: Optional[dict] = {},  # Model specific template
) -> str:
    # (end tokens, structure, etc)
    # @TODO Pass these in from UI model_configs.json (values found in config.json of HF model card)
    BOS = template["BOS"] or ""
    EOS = template["EOS"] or ""
    B_INST = template["B_INST"] or ""
    E_INST = template["E_INST"] or ""
    B_SYS = template["B_SYS"] or ""
    E_SYS = template["E_SYS"] or ""

    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

        if i == 0:
            # make sure system prompt is included at the start
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            # end previous user-assistant interaction
            string_messages[-1] += f" {EOS}"
            # no need to include system prompt
            str_message = f"{BOS} {B_INST} "

        # include user message content
        str_message += f"{user_message.content} {E_INST}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


# Format the prompt for completion
# @TODO Could also use: from llama_index.llms.llama_cpp.llama_utils import completion_to_prompt
def _completion_to_prompt(
    completion: Optional[str] = "",
    system_prompt: Optional[str] = None,
    template_str: Optional[str] = None,  # Model specific template
):
    if not system_prompt or len(system_prompt.strip()) == 0:
        system_prompt = DEFAULT_SYSTEM_MESSAGE

    # print(
    #     f"\nprompt:\n{completion}\n\nsystem_prompt:\n{system_prompt}template:\n{template_str}",
    #     flush=True,
    # )

    # Format to default spec if no template supplied
    if not template_str:
        return f"{system_prompt.strip()} {completion.strip()}"

    # Format to specified template
    prompt_str = template_str.replace("{prompt}", completion.strip())
    completion_str = prompt_str.replace("{system_message}", system_prompt.strip())
    return completion_str


# Load a local model and return an OpenAI api compatible class
def create(
    path_to_model: str,
    model_name: str,
    mode: str,
    init_settings: classes.LoadTextInferenceInit,  # init settings
    generate_settings: classes.LoadTextInferenceCall,  # generation settings
    # callback_manager: CallbackManager = None,  # Optional, debugging
) -> Llama_CPP_SERVER:
    n_ctx = init_settings.n_ctx or classes.DEFAULT_CONTEXT_WINDOW
    if n_ctx <= 0:
        n_ctx = classes.DEFAULT_CONTEXT_WINDOW
    seed = init_settings.seed
    n_threads = init_settings.n_threads  # None means auto calc
    if n_threads == -1:
        n_threads = None

    model_init_kwargs = {
        # "--device": "CUDA0",
        "--n_gpu_layers": init_settings.n_gpu_layers,
        "--no_mmap": not init_settings.use_mmap,
        "--mlock": init_settings.use_mlock,
        # "--cache-type-k": init_settings.cache_type_k, # @TODO implement this
        # "--cache-type-v": init_settings.cache_type_v, # @TODO implement this
        # "f16_kv": init_settings.f16_kv, # @TODO deprecate this
        "--seed": seed,
        "--ctx-size": n_ctx,  # 0=loaded from model
        "--batch-size": init_settings.n_batch,  # default: 2048
        "--threads": n_threads or -1,
        "--threads-batch": n_threads or -1,
        "--no-kv-offload": not init_settings.offload_kqv,
        # "torch_dtype": "auto",  # if using CUDA (reduces memory usage) # @TODO deprecate
    }

    llm = Llama_CPP_SERVER(
        # Provide a url to download a model from
        model_url=None,
        # Or, you can set the path to a pre-downloaded model instead of model_url
        model_path=path_to_model,
        # Friendly name
        model_name=model_name,
        # kwargs to pass when generating text
        generate_kwargs=generate_settings,
        # kwargs to pass when loading the model
        model_init_kwargs=model_init_kwargs,
        # Transform inputs into model specific format
        messages_to_prompt=_messages_to_prompt,
        completion_to_prompt=_completion_to_prompt,
        # @TODO Implement this for debugging or monitoring tooling
        # callback_manager=callback_manager,
        debug=True,
        # CHAT_MODES (instruct, etc)
        mode=mode,
    )
    return llm
