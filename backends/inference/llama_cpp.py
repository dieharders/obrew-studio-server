import os
import asyncio
import json
from typing import List, Optional
from core import common
from inference.helpers import (
    completion_to_prompt,
    make_chunk_payload,
    sanitize_kwargs,
)
from inference.classes import (
    ChatMessage,
    InferenceRequest,
    LoadTextInferenceCall,
    LoadTextInferenceInit,
    ChatHistory,
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
        model_id: str,  #  id of model in config
        active_role: str,  # ACTIVE_ROLES
        response_mode: str,  # CHAT_MODES
        raw: bool,  # user can send manually formatted messages
        # template converts messages to prompts
        message_format: Optional[dict] = {},
        verbose=False,
        debug=False,  # Show logs
        model_init_kwargs: LoadTextInferenceInit = None,  # kwargs to pass when loading the model
        generate_kwargs: (
            LoadTextInferenceCall | InferenceRequest
        ) = None,  # kwargs to pass when generating text
    ):
        # Set args
        n_ctx = model_init_kwargs.n_ctx or DEFAULT_CONTEXT_WINDOW
        if n_ctx <= 0:
            n_ctx = DEFAULT_CONTEXT_WINDOW
        n_threads = model_init_kwargs.n_threads
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
            "--cache-type-k": model_init_kwargs.cache_type_k,
            "--cache-type-v": model_init_kwargs.cache_type_v,
        }
        if n_threads != None:
            init_kwargs["--threads"] = n_threads
            init_kwargs["--threads-batch"] = n_threads
        # Assign vars
        self.max_empty = 100
        self.chat_history = None
        self.process = None
        self.task_logging = None
        self.prompt_template = None  # structures the llm thoughts (thinking)
        self.message_format = message_format
        self.request_queue = asyncio.Queue()
        self.response_mode = response_mode
        self.active_role = active_role
        self.raw = raw
        self.model_url = model_url
        self.model_name = model_name or "chatbot"  # human friendly name for display
        self.model_id = model_id
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
        stopwords = settings.stop
        # Create args to pass to .exe
        kwargs = {
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
        if stopwords:
            # Never use an empty string like [""] or empty array []
            kwargs.update({"--reverse-prompt": stopwords})
        if grammar:
            kwargs.update({"--grammar": grammar})
        self._generate_kwargs = kwargs

    # Remove request from queue
    async def complete_request(self):
        """Remove the last request and complete the task."""
        self.request_queue.get_nowait()
        self.request_queue.task_done()  # Signal the end of requests
        await self.request_queue.join()  # Wait for all tasks to complete
        return

    # Create a cli instance and load a previous conversation (only needed for chat convo)
    # @TODO This is yet to be implemented
    async def load_cached_chat(
        self,
        # contain at minimum the system_message
        chat_history: Optional[ChatHistory] = None,
    ):
        """
        Starts the llama.cpp cli subprocess for chat conversation.
        """

        try:
            if not chat_history:
                return
            else:
                self.chat_history = chat_history
            return
        except Exception as e:
            print(f"{common.PRNT_LLAMA} Error occurred: {e}")

    # Shutdown instance
    def unload(self):
        try:
            # Cleanup any ongoing processes
            if self.task_logging:
                self.task_logging.cancel()
            if self.process:
                print(
                    f"{common.PRNT_LLAMA} Shutting down llama.cpp cli process.",
                    flush=True,
                )
                self.process.kill()

        except ProcessLookupError as e:
            print(f"{common.PRNT_LLAMA_LOG} Could not find process to kill: {e}")
        except Exception as e:
            print(f"{common.PRNT_LLAMA_LOG} Error occurred: {e}")
        finally:
            self.process = None
            self.task_logging = None

    async def read_logs(self):
        async for line in self.process.stderr:
            # Print logs in real-time
            print(f"{common.PRNT_LLAMA_LOG}", line.decode("utf-8").strip())

    # Load a previous chat conversation
    # @TODO Not implemented, needs chat_to_completions(chat_history) to convert conversation to string
    async def load_chat(self, chat_history: List[ChatMessage] | None):
        if not chat_history:
            return
        return

    # User can pause Ai and add more input, keeps conn open.
    # @TODO Yet to be implemented
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
        return

    # Send multi-turn messages by role. Message does not require formatting.
    # Cannot reload chat history from cache.
    async def text_chat(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
    ):
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
                "--no-warmup",  # skip warming up the model with an empty run
                "-cnv",  # conversation mode
            ]

            # Configure args
            cmd_args.append("--in-prefix")
            cmd_args.append("")
            cmd_args.append("--in-suffix")
            cmd_args.append("")
            # Sets system message when using "-cnv" mode
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
            if override_args:
                merged_args.update(override_args)  # Add overrides
            sanitized = sanitize_kwargs(kwargs=merged_args)
            cmd_args.extend(sanitized)

            # Start process
            if not self.process:
                print(
                    f"{common.PRNT_LLAMA} Starting llama.cpp cli ...\nWith chat command: {cmd_args}"
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
            command = f"{prompt.strip()}\\"
            print(f"{common.PRNT_LLAMA} Generating chat with command: {command}")
            self.process.stdin.write(command.encode("utf-8") + b"\n")
            await self.process.stdin.drain()

            if stream:
                # Stream response as SSE
                index = 0
                num_empty = 0
                while True:
                    if not self.process:
                        break
                    # Read and parse line as string
                    index += 1
                    line = await self.process.stdout.read(4)
                    line = line.decode("utf-8")
                    # Bail/skip on empty line
                    if line.strip() == ">":
                        if index == 1:
                            continue
                        break
                    # Bail on empty
                    if line:
                        num_empty = 0
                    if line == "":
                        num_empty += 1
                        if num_empty > self.max_empty:
                            break
                        continue

                    payload = make_chunk_payload(line)
                    yield json.dumps(payload)  # SSE format expects json
            else:
                # Return entire result in one response
                content = ""
                index = 0
                num_empty = 0
                marker_num = 0
                while True:
                    if not self.process:
                        break
                    # Read and parse line as string
                    index += 1
                    line = await self.process.stdout.read(1)
                    line = line.decode("utf-8")
                    # Bail if second > found
                    if line.strip().startswith(">"):
                        if marker_num > 0:
                            break
                        marker_num += 1
                    # Bail/skip on empty line
                    if line:
                        num_empty = 0
                    if line == "":
                        num_empty += 1
                        if num_empty > self.max_empty:
                            break
                        continue

                    content += line
                content = content.lstrip(">").rstrip("\r\n>").strip()
                payload = make_chunk_payload(content)
                yield payload

            # Finished - cleanup
            if self.task_logging:
                self.task_logging.cancel()
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled.")
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error querying llama.cpp: {e}")
            raise Exception(f"Failed to query llama.cpp: {e}")
        except Exception as e:
            print(f"{common.PRNT_LLAMA} Some error occurred: {e}")

    # Predict the rest of the prompt
    # FYI, if we want to load prev conversation from cache and continue from there, most likely must use completion since -cnv mode makes it difficult to reload and manage chat history.
    async def text_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
    ):
        try:
            # If format type provided pass input unchanged, llama.cpp will handle it?
            formatted_prompt = prompt.strip()
            # Format prompt if none specified
            if not self.raw:
                formatted_prompt = completion_to_prompt(
                    user_message=prompt,
                    system_message=system_message,
                    template=self.message_format,
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
            if override_args:
                merged_args.update(override_args)  # Add overrides
            sanitized = sanitize_kwargs(kwargs=merged_args)
            cmd_args.extend(sanitized)

            # Start process
            print(
                f"{common.PRNT_LLAMA} Starting llama.cpp cli ...\nWith completion command: {cmd_args}"
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
            eos_token = "[end of text]"
            if stream:
                num_empty = 0
                while True:
                    if not self.process:
                        break
                    line = await self.process.stdout.read(1)
                    line = line.decode("utf-8")
                    eos_index = line.find(eos_token)

                    if eos_index != -1:
                        break

                    # Bail if if we find > (could be error since this shouldnt show up)
                    if line.strip().startswith(">"):
                        break
                    # Bail/skip on empty line
                    if line:
                        num_empty = 0
                    if line == "":
                        num_empty += 1
                        if num_empty > self.max_empty:
                            break
                        continue

                    # Send tokens
                    payload = make_chunk_payload(line)
                    yield json.dumps(payload)  # SSE format expects json
            else:
                # Return entire result in one response
                content = ""
                num_empty = 0
                while True:
                    if not self.process:
                        break
                    # Read and parse line as string
                    line = await self.process.stdout.read(1)
                    line = line.decode("utf-8")
                    # Bail if if we find > (could be error since this shouldnt show up)
                    if line.strip().startswith(">"):
                        break
                    # Bail on EOS token
                    if line.find(eos_token) != -1:
                        content += line
                        break
                    # Bail/skip on empty line
                    if line:
                        num_empty = 0
                    if line == "":
                        num_empty += 1
                        if num_empty > self.max_empty:
                            break
                        continue
                    # Add line to text result
                    content += line
                content = content.strip().rstrip(eos_token).rstrip()
                payload = make_chunk_payload(content)
                yield payload

            # Finished - cleanup
            if self.task_logging:
                self.task_logging.cancel()
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled.")
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error querying llama.cpp: {e}")
            raise Exception(f"Failed to query llama.cpp: {e}")
        except Exception as e:
            print(f"{common.PRNT_LLAMA} Some error occurred: {e}")
