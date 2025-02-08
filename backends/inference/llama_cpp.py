import os
import asyncio
import json
from typing import List, Optional, Sequence
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from core import common
from inference.classes import (
    InferenceRequest,
    LoadTextInferenceCall,
    LoadTextInferenceInit,
    CHAT_MODES,
    DEFAULT_CONTEXT_WINDOW,
)

# More templates found here: https://github.com/run-llama/llama_index/blob/main/llama_index/prompts/default_prompts.py
DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
"""

DEFAULT_PROMPT_FORMAT = """### HUMAN:
{{prompt}}

### RESPONSE:"""


# Convert input to the target prompt format for a given model.
# Note completions dont utilize system message (b/c neither do Instruct type models) so we combine it with input msg if present.
def _apply_prompt_template(
    input_message: Optional[str] = "",
    system_message: Optional[str] = None,
    template_str: Optional[str] = None,  # Model specific template
):
    if template_str:
        txt = input_message.strip()
        # @TODO Do this for /completions: template_str.replace("{{system_message}}", txt) instead ?
        # ...maybe make sep func for this

        # @TODO Do this instead for chat since system_message is sent seperatly
        if system_message:
            txt = f"{system_message.strip()}\n\n{input_message.strip()}"
        # Format to specified template
        return template_str.replace("{{prompt}}", txt)
    else:
        # Dont format if no template supplied
        if system_message:
            return f"{system_message.strip()}\n{input_message.strip()}"
        return f"{input_message.strip()}"


# https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
def _format_chat_to_prompt(
    user_message, chat_history=[], system_message=DEFAULT_SYSTEM_MESSAGE
):
    """Manually formats a chat conversation like `--chat-template`."""

    # @TODO Use if chat_history is provided
    formatted_history = "\n".join(
        f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history
    )

    return f"""
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """


def _sanitize_kwargs(kwargs: dict) -> list[str]:
    arr = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value == True:
                arr.append(key)
            else:
                pass
        else:
            arr.extend([f"{key}", str(value)])
    return arr


# Run a llama.cpp cli binary
# https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#common-options
class LLAMA_CPP:
    def __init__(
        self,
        model_url: str,  # url to download from
        model_path: str,  # file path
        model_name: str,  # friendly name
        mode: str,
        prompt_format: Optional[str] = None,
        verbose=False,
        debug=False,
        model_init_kwargs: dict = None,
        generate_kwargs: dict = None,
        priority: int = 0,  # set process priority (0=default, 1, 2, 3)
    ):
        self.process = None
        self.task_logging = None
        self.promptTemplate = None
        self.messageFormat = None
        self.request_queue = asyncio.Queue()
        self.message_format_type = None
        self.prompt_format = prompt_format
        self.mode = mode
        self.model_url = model_url
        self.model_name = model_name or "chatbot"  # human friendly name for display
        self.verbose = verbose
        self.debug = debug
        self.priority = priority
        self.model_path = model_path  # text-models/your-model.gguf
        self.model_init_kwargs = model_init_kwargs
        self._generate_kwargs = generate_kwargs  # proxy
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
    def generate_kwargs(self, settings: InferenceRequest):
        grammar = settings.grammar
        self.promptTemplate = settings.promptTemplate
        self.messageFormat = settings.messageFormat
        # Create args to pass to .exe
        kwargs = {
            # "stream": settings.stream, # @TODO Should implement for non-stream on route level
            "--ctx-size": self.model_init_kwargs.get("--ctx-size"),
            "--reverse-prompt": settings.stop,  # !Never use an empty string like [""]
            "--model": settings.model,
            "--batch-size": self.model_init_kwargs.get("--batch-size"),
            "--mirostat-ent": settings.mirostat_tau,  # (tau) default 5.0
            # "--tfs-z": settings.tfs_z, # @TODO deprecate
            "--top-k": settings.top_k,
            "--top-p": settings.top_p,
            "--min-p": settings.min_p,
            "--repeat-penalty": settings.repeat_penalty,
            "--presence-penalty": settings.presence_penalty,
            "--frequency-penalty": settings.frequency_penalty,
            "--temp": settings.temperature,
            "--seed": self.model_init_kwargs.get("--seed"),
            "--n-predict": settings.max_tokens,
        }
        if grammar:
            kwargs.update({"--grammar": grammar})
        self._generate_kwargs = kwargs

    # Shutdown instance
    def unload(self):
        try:
            # Cleanup any ongoing processes
            if self.process:
                print(
                    f"{common.PRNT_LLAMA} Shutting down llama.cpp cli process.",
                    flush=True,
                )
                self.process.kill()

        except ProcessLookupError as e:
            print(f"{common.PRNT_LLAMA_LOG} Could not find process to kill: {e}")
        except e:
            print(f"{common.PRNT_LLAMA_LOG} Error occurred: {e}")
        finally:
            self.process = None
            self.task_logging = None

    async def read_logs(self):
        async for line in self.process.stderr:
            print(
                f"{common.PRNT_LLAMA_LOG}", line.decode("utf-8").strip()
            )  # Print logs in real-time

    # Create a cli instance and load in a model
    # @TODO This should save all necessary values to the class object for subsequent requests to use during generation.
    async def load_model(
        self,
        mode: CHAT_MODES = CHAT_MODES.CHAT.value,
        message_format_type=None,  # "llama2"
        system_message=None,
    ):
        """
        Starts the llama.cpp cli using subprocess.
        """

        try:
            # Create arguments for starting server
            cmd_args = [
                self.BINARY_PATH,
                "--model",
                self.model_path,
                "--no-display-prompt",
                "--no-context-shift",
                "--simple-io",
                "--multiline-input",  # dont have to type "/" to add a new line
            ]
            if mode == CHAT_MODES.CHAT.value:
                cmd_args.append("-cnv")  # conversation mode (chat)
                if message_format_type:
                    # if not used then gotten from model (or manually format)
                    cmd_args.append("--chat-template")
                    cmd_args.append(message_format_type)
                    self.message_format_type = message_format_type
                else:
                    cmd_args.append("--in-prefix")
                    cmd_args.append("")
                    cmd_args.append("--in-suffix")
                    cmd_args.append("")
                if system_message:
                    # Changes system message when using "-cnv" mode
                    cmd_args.append("--prompt")
                    cmd_args.append(system_message)  # @TODO Get from system_message
            elif mode == CHAT_MODES.COLLAB.value:
                # User can pause Ai and add more input, keeps conn open.
                cmd_args.append("--interactive")
                # "--ignore-eos -n -1",  # infinite response
            else:
                # Perform instruct op if no matches for specific mode
                return

            # Sanitize values
            sanitized = _sanitize_kwargs(kwargs=self.model_init_kwargs)
            cmd_args.extend(sanitized)

            # @TODO We can move everything from here down and make this (and args creation logic) to a helper func and call in each inference func. That way we only actually mount the model at inference request.

            # Start process
            print(
                f"{common.PRNT_LLAMA} Starting llama.cpp cli ...\nWith command: {cmd_args}"
            )

            self.process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                bufsize=0,
            )
            # Read logs
            if self.debug:
                self.task_logging = asyncio.create_task(self.read_logs())

        except Exception as e:
            print(f"{common.PRNT_LLAMA} Error starting the llama.cpp: {e}")
            raise Exception(f"Error starting llama.cpp: {e}")

    # Send multi-turn messages by role
    # @TODO do something with message_format
    async def text_chat(
        self,
        prompt: str,
        system_message: str = None,
    ):
        try:
            # Format command for llama-cli
            # @TODO Incorporate with model_init_kwargs
            # kwargs = " ".join(f"{k} {v}" for k, v in self.generate_kwargs.items())

            # Format prompt for chat
            # formatted_prompt = json.dumps(
            #     {
            #         "messages": [
            #             {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            #             {"role": "user", "content": prompt},
            #         ]
            #     }
            # )

            # If format type provided pass input unchanged, llama.cpp will handle it?
            formatted_prompt = prompt
            # Format prompt if none specified
            if self.message_format_type == None:
                formatted_prompt = _apply_prompt_template(
                    input_message=prompt,
                    system_message=system_message,
                    template_str=self.messageFormat,
                )

            # Send command to llama-cli
            command = f"{formatted_prompt}\\"
            print(f"{common.PRNT_LLAMA} Generating chat with command: {command}")
            self.process.stdin.write(command.encode("utf-8") + b"\n")
            await self.process.stdin.drain()

            # Stream response as SSE
            index = 0
            while True:
                line = await self.process.stdout.read(4)
                line = line.decode("utf-8")

                if line.strip() == ">":
                    if index == 0:
                        continue
                    else:
                        break

                # print(f"line:{line}", flush=True)
                index += 1
                payload = {
                    "event": "GENERATING_TOKENS",
                    "data": line,
                }
                yield json.dumps(payload)  # SSE format

            # Finished - cleanup
            self.task_logging.cancel()
            # Cleanup queue (if used for this)
            # self.request_queue.get_nowait()
            # self.request_queue.task_done()  # Signal the end of requests
            # await self.request_queue.join()  # Wait for all tasks to complete
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled.")
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error querying the llama.cpp: {e}")
            raise Exception(f"Error querying the llama.cpp: {e}")
        except e:
            print(f"{common.PRNT_LLAMA} Some error occurred: {e}")

    # Predict the rest of the prompt
    async def text_completion(
        self,
        prompt: str,
        system_message: str = None,
    ):
        try:
            # Format command for llama-cli
            # @TODO Incorporate with model_init_kwargs
            # kwargs = " ".join(f"{k} {v}" for k, v in self.generate_kwargs.items())

            # If format type provided pass input unchanged, llama.cpp will handle it?
            formatted_prompt = prompt
            # Format prompt if none specified
            if self.message_format_type == None:
                formatted_prompt = _apply_prompt_template(
                    input_message=prompt,
                    system_message=system_message,
                    template_str=self.messageFormat,
                )

            # Create arguments
            cmd_args = [
                self.BINARY_PATH,
                "--model",
                self.model_path,
                "--no-display-prompt",
                "--no-context-shift",
                "--simple-io",
                "--multiline-input",  # dont have to type "/" to add a new line
                "--no-warmup",  # skip warming up the model with an empty run
                "--prompt",
                formatted_prompt,
            ]

            # Sanitize values
            sanitized = _sanitize_kwargs(kwargs=self.model_init_kwargs)
            cmd_args.extend(sanitized)

            # Start process
            print(
                f"{common.PRNT_LLAMA} Starting llama.cpp cli ...\nWith command: {cmd_args}"
            )

            # Send command to llama-cli
            self.process = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                bufsize=0,
            )
            # Read logs
            if self.debug:
                self.task_logging = asyncio.create_task(self.read_logs())

            # Send command to llama-cli
            await self.process.stdin.drain()

            # Stream response as SSE
            while True:
                line = await self.process.stdout.read(14)
                line = line.decode("utf-8")
                eos_token = "[end of text]"  # @TODO Do we need to figure this out for each model?
                eos_index = line.find(eos_token)

                # print(f"line:{line}", flush=True)

                if eos_index != -1:
                    break

                payload = {
                    "event": "GENERATING_TOKENS",
                    "data": line,
                }
                yield json.dumps(payload)  # SSE format

            # Finished - cleanup
            self.task_logging.cancel()
            self.request_queue.get_nowait()
            self.request_queue.task_done()  # Signal the end of requests
            await self.request_queue.join()  # Wait for all tasks to complete
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled.")
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error querying the llama.cpp: {e}")
            raise Exception(f"Error querying the llama.cpp: {e}")
        except e:
            print(f"{common.PRNT_LLAMA} Some error occurred: {e}")


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


# Load a local model and return an OpenAI api compatible class
def create(
    path_to_model: str,
    model_name: str,
    mode: str,
    init_settings: LoadTextInferenceInit,  # init settings
    generate_settings: LoadTextInferenceCall,  # generation settings
) -> LLAMA_CPP:
    n_ctx = init_settings.n_ctx or DEFAULT_CONTEXT_WINDOW
    if n_ctx <= 0:
        n_ctx = DEFAULT_CONTEXT_WINDOW
    n_threads = init_settings.n_threads  # None means auto calc
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
        "--seed": init_settings.seed,
        "--ctx-size": n_ctx,  # 0=loaded from model
        "--batch-size": init_settings.n_batch,
        "--no-kv-offload": not init_settings.offload_kqv,
        # "torch_dtype": "auto",  # if using CUDA (reduces memory usage) # @TODO deprecate
    }
    if n_threads != None:
        model_init_kwargs["--threads"] = n_threads
        model_init_kwargs["--threads-batch"] = n_threads

    # @TODO Cant we do this instead of calling create() ?
    llm = LLAMA_CPP(
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
        # Show logs
        debug=True,
        # CHAT_MODES (instruct, etc)
        mode=mode,
    )
    return llm
