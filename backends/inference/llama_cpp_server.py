import time
import os
import httpx
import asyncio
import subprocess
import requests
import json
import atexit
from fastapi import HTTPException
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


# https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
def format_chat_prompt(
    user_message, chat_history=[], system_message=DEFAULT_SYSTEM_MESSAGE
):
    """Manually formats a chat conversation like `--chat-template`."""

    # @TODO Use when is_chat or if chat_history is provided
    formatted_history = "\n".join(
        f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history
    )

    # @TODO Fix this ...
    # llama2?
    # <|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant

    return f"""
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """
    # return f"""
    #     [INST] <<SYS>>
    #     {system_message}
    #     <</SYS>>
    #     {formatted_history}
    #     User: {user_message}
    #     Assistant:
    #     """


# Run a llama.cpp server binary wrapped in OpenAI api compatible-ish class
# https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#common-options
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
        self.request_queue = asyncio.Queue()
        self.message_format_type = None
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
            os.getcwd(), BINARY_BASE_PATH, BINARY_FOLDER, "llama-cli.exe"
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
        grammar = settings.get("grammar")
        kwargs = {
            # "stream": settings.get("stream"), # @TODO Need to implement for non-stream on route level
            # "stop": settings.get("stop"),  # !Never use an empty string like [""] @TODO Implement using "Reverse Prompt"
            "--model": settings.get("model"),
            # "--batch-size": 2048, # @TODO get from UI
            "--mirostat-ent": settings.get("mirostat_tau"),  # (tau) default 5.0
            # "--tfs-z": settings.get("tfs_z"), # @TODO deprecate
            "--top-k": settings.get("top_k"),
            "--top-p": settings.get("top_p"),
            "--min-p": settings.get("min_p"),
            "--repeat-penalty": settings.get("repeat_penalty"),
            "--presence-penalty": settings.get("presence_penalty"),
            "--frequency-penalty": settings.get("frequency_penalty"),
            "--temp": settings.get("temperature"),
            "--seed": self.model_init_kwargs.get("--seed"),
            "--predict": common.calc_max_tokens(
                settings.get("max_tokens"), n_ctx, self.mode
            ),
        }
        if grammar:
            kwargs.update({"--grammar": grammar})
        self._generate_kwargs = kwargs

    # Shutdown server
    def unload(self):
        if self.process:
            print(
                f"{common.PRNT_LLAMA} Shutting down llama.cpp cli process.",
                flush=True,
            )
            self.process.kill()

    async def read_logs(self):
        async for line in self.process.stderr:
            print("[LOG]", line.decode("utf-8").strip())  # Print logs in real-time

    # Create a cli instance and load in a model
    async def load_model(
        self,
        is_chat=False,
        message_format_type=None,  # "llama2"
        system_message=DEFAULT_SYSTEM_MESSAGE,
    ):
        """
        Starts the llama.cpp cli using subprocess.
        """

        try:
            # Create arguments for starting server
            cmd_args = [
                self.BINARY_PATH,
                # "--no-warmup",  # skip warming up the model with an empty run
                "--interactive",  # User can pause Ai and add more input, keeps conn open.
                # "--ignore-eos -n -1",  # infinite response
                "--multiline-input",
                "--model",
                self.model_path,
            ]
            if message_format_type:
                # if not used then gotten from model (or manually format)
                cmd_args.append("--chat-template")
                cmd_args.append(message_format_type)
            self.message_format_type = message_format_type
            if is_chat:
                cmd_args.append("-cnv")  # conversation mode (chat)
                # Changes system message when using "-cnv" mode
                cmd_args.append("--prompt")
                cmd_args.append(system_message),  # @TODO Get from system_message
            else:
                cmd_args.append("-no-cnv")

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
                f"{common.PRNT_LLAMA} Starting llama.cpp cli ...\nWith command: {cmd_args}"
            )

            # Close if instance already exists
            # @TODO Could re-use process if the prev and loading model are the same
            if self.process:
                self.process.kill()
            self.process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            # Read logs
            if self.debug:
                asyncio.create_task(self.read_logs())

        except Exception as e:
            print(f"{common.PRNT_LLAMA} Error starting the llama.cpp: {e}")
            raise Exception(f"Error starting llama.cpp: {e}")

    # Predict the rest of the prompt
    # @TODO do something with message_format
    async def text_completion(
        self,
        prompt: str,
        system_message: str = DEFAULT_SYSTEM_MESSAGE,
        is_chat=False,
    ):
        try:
            # Format command for llama-cli
            # @TODO Handle "messages": [] for chat mode v1/chat/completions
            kwargs = " ".join(f"{k} {v}" for k, v in self.generate_kwargs.items())
            # Format prompt for chat
            formatted_prompt = json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ]
                }
            )
            # Format prompt if none provided
            if self.message_format_type == None:
                formatted_prompt = format_chat_prompt(
                    user_message=prompt, system_message=system_message
                )
            command = f"--prompt {formatted_prompt} --no-context-shift --no-display-prompt {kwargs}"
            # Send command to llama-cli
            print(f"{common.PRNT_LLAMA} Generating with command: {command}")
            self.process.stdin.write(command.encode("utf-8") + b"\n")
            await self.process.stdin.drain()
            # Stream response as SSE
            async for line in self.process.stdout:
                payload = {
                    "event": "GENERATING_TOKENS",
                    "data": line.decode("utf-8").strip(),
                }
                yield json.dumps(payload)  # SSE format
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error querying the llama.cpp: {e}")
            raise Exception(f"Error querying the llama.cpp: {e}")


# Convert structured chat conversation to prompt (str)
# @TODO Could also use: from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt
def _messages_to_prompt(
    messages: Sequence[ChatMessage],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
    template: Optional[dict] = {},  # Model specific template
) -> str:
    # (end tokens, structure, etc)
    # @TODO Pass these in from UI model_configs.json (values found in config.json of HF model card)
    BOS = template["BOS"] or ""  # begin string
    EOS = template["EOS"] or ""  # end of string
    B_INST = template["B_INST"] or ""  # begin instruction (user prompt)
    E_INST = template["E_INST"] or ""  # end instruction (user prompt)
    B_SYS = template["B_SYS"] or ""  # begin system instruction
    E_SYS = template["E_SYS"] or ""  # end system instruction

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


# Convert input to completion prompt
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
    n_gpu_layers = init_settings.n_gpu_layers
    if init_settings.n_gpu_layers == -1:
        n_gpu_layers = 100  # all

    model_init_kwargs = {
        # "--device": "CUDA0",
        "--n_gpu_layers": n_gpu_layers,
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
