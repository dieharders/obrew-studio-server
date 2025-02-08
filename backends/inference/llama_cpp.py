import os
import asyncio
import json
from typing import Literal, Optional
from inference.helpers import apply_prompt_template, sanitize_kwargs
from core import common
from inference.classes import (
    InferenceRequest,
    LoadTextInferenceCall,
    LoadTextInferenceInit,
    CHAT_MODES,
    DEFAULT_CONTEXT_WINDOW,
)


# https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#common-options
class LLAMA_CPP:
    """Run a llama.cpp cli binary"""

    def __init__(
        self,
        model_url: str,  # Provide a url to download a model from
        model_path: str,  # Or, you can set the path to a pre-downloaded model file instead of model_url
        model_name: str,  # Friendly name
        mode: Literal[CHAT_MODES.CHAT, CHAT_MODES.COLLAB, CHAT_MODES.INSTRUCT],
        prompt_format: Optional[str] = None,
        message_format: Optional[str] = None,
        verbose=False,
        debug=False,  # Show logs
        model_init_kwargs: LoadTextInferenceInit = None,  # kwargs to pass when loading the model
        generate_kwargs: LoadTextInferenceCall = None,  # kwargs to pass when generating text
    ):
        # Set args
        n_ctx = model_init_kwargs.n_ctx or DEFAULT_CONTEXT_WINDOW
        if n_ctx <= 0:
            n_ctx = DEFAULT_CONTEXT_WINDOW
        n_threads = model_init_kwargs.n_threads  # None means auto calc
        n_gpu_layers = model_init_kwargs.n_gpu_layers
        if model_init_kwargs.n_gpu_layers == -1:
            n_gpu_layers = 100  # all

        init_kwargs = {
            "--n_gpu_layers": n_gpu_layers,
            "--no_mmap": not model_init_kwargs.use_mmap,
            "--mlock": model_init_kwargs.use_mlock,
            "--seed": model_init_kwargs.seed,
            "--ctx-size": n_ctx,  # 0=loaded from model
            "--batch-size": model_init_kwargs.n_batch,
            "--no-kv-offload": not model_init_kwargs.offload_kqv,
            # "--device": "CUDA0", # optional, target a specific device
            # "--cache-type-k": init_settings.cache_type_k, # @TODO implement this
            # "--cache-type-v": init_settings.cache_type_v, # @TODO implement this
        }
        if n_threads != None:
            init_kwargs["--threads"] = n_threads
            init_kwargs["--threads-batch"] = n_threads
        # Assign vars
        self.process = None
        self.task_logging = None
        self.prompt_template = None
        self.message_format = message_format
        self.request_queue = asyncio.Queue()
        self.message_format_type = None
        self.prompt_format = prompt_format
        self.mode = mode
        self.model_url = model_url
        self.model_name = model_name or "chatbot"  # human friendly name for display
        self.verbose = verbose
        self.debug = debug
        self.model_path = model_path  # text-models/your-model.gguf
        self.model_init_kwargs = init_kwargs
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
        self.prompt_template = settings.promptTemplate
        self.message_format = settings.messageFormat
        # Create args to pass to .exe
        kwargs = {
            "--reverse-prompt": settings.stop,  # Never use an empty string like [""]
            "--mirostat-ent": settings.mirostat_tau,  # (tau) default 5.0
            "--top-k": settings.top_k,
            "--top-p": settings.top_p,
            "--min-p": settings.min_p,
            "--repeat-penalty": settings.repeat_penalty,
            "--presence-penalty": settings.presence_penalty,
            "--frequency-penalty": settings.frequency_penalty,
            "--ctx-size": self.model_init_kwargs.get("--ctx-size"),
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

    # User can pause Ai and add more input, keeps conn open.
    async def text_collab(self):
        # Create arguments for starting server
        cmd_args = [
            self.BINARY_PATH,
            "--model",
            self.model_path,
            "--no-display-prompt",
            "--no-context-shift",
            "--simple-io",
            "--multiline-input",  # dont have to type "/" to add a new line
            "--interactive",
            "--ignore-eos -n -1",  # infinite response
        ]
        # @TODO launch binary and pass args

    # Send multi-turn messages by role
    async def text_chat(
        self,
        prompt: str,
        system_message: str = None,
    ):
        try:
            # If format type provided pass input unchanged, llama.cpp will handle it?
            formatted_prompt = prompt
            # Format prompt if none specified
            if self.message_format_type == None:
                formatted_prompt = apply_prompt_template(
                    input_message=prompt,
                    system_message=system_message,
                    template_str=self.message_format,
                )

            # Create arguments for starting server
            cmd_args = [
                self.BINARY_PATH,
                "--model",
                self.model_path,
                "--no-display-prompt",
                "--no-context-shift",
                "--simple-io",
                "--multiline-input",  # dont have to type "/" to add a new line
                "--no-warmup",  # skip warming up the model with an empty run
                "-cnv",  # conversation mode
            ]

            # Configure args
            if self.message_format_type:
                # if not used then get from model (or manually format)
                cmd_args.append("--chat-template")
                cmd_args.append(self.message_format_type)
            else:
                cmd_args.append("--in-prefix")
                cmd_args.append("")
                cmd_args.append("--in-suffix")
                cmd_args.append("")
            # Changes system message when using "-cnv" mode
            if system_message:
                cmd_args.append("--prompt")
                cmd_args.append(system_message)
            # Add stop words
            if self.generate_kwargs.get("--reverse-prompt"):
                cmd_args.append("--reverse-prompt")
                cmd_args.append(self.generate_kwargs.get("--reverse-prompt"))

            # Sanitize args
            merged_args = self.model_init_kwargs.copy()
            merged_args.update(self.generate_kwargs)
            sanitized = sanitize_kwargs(kwargs=merged_args)
            cmd_args.extend(sanitized)

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
            # Cleanup queue
            self.request_queue.get_nowait()
            self.request_queue.task_done()  # Signal the end of requests
            await self.request_queue.join()  # Wait for all tasks to complete
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled.")
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error querying llama.cpp: {e}")
            raise Exception(f"Error querying llama.cpp: {e}")
        except e:
            print(f"{common.PRNT_LLAMA} Some error occurred: {e}")

    # Predict the rest of the prompt
    async def text_completion(
        self,
        prompt: str,
        system_message: str = None,
    ):
        try:
            # If format type provided pass input unchanged, llama.cpp will handle it?
            formatted_prompt = prompt
            # Format prompt if none specified
            if self.message_format_type == None:
                formatted_prompt = apply_prompt_template(
                    input_message=prompt,
                    system_message=system_message,
                    template_str=self.message_format,
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
            # Add stop words
            if self.generate_kwargs.get("--reverse-prompt"):
                cmd_args.append("--reverse-prompt")
                cmd_args.append(self.generate_kwargs.get("--reverse-prompt"))

            # Sanitize args
            merged_args = self.model_init_kwargs.copy()
            merged_args.update(self.generate_kwargs)
            sanitized = sanitize_kwargs(kwargs=merged_args)
            cmd_args.extend(sanitized)

            # Start process
            print(
                f"{common.PRNT_LLAMA} Starting llama.cpp cli ...\nWith command: {cmd_args}"
            )

            # Send command to llama-cli
            # @TODO Incorporate with model_init_kwargs
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
            print(f"{common.PRNT_LLAMA} Error querying llama.cpp: {e}")
            raise Exception(f"Error querying llama.cpp: {e}")
        except e:
            print(f"{common.PRNT_LLAMA} Some error occurred: {e}")
