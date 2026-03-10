import os
import json
import codecs
import signal
import asyncio
import subprocess
import platform
from asyncio.subprocess import Process
from fastapi import Request
from typing import List, Optional
from core import common
from inference.helpers import (
    FEEDING_PROMPT,
    GENERATING_TOKENS,
    cleanup_temp_file,
    write_prompt_to_temp_file,
    completion_to_prompt,
    event_payload,
    token_payload,
    content_payload,
    sanitize_kwargs,
    inference_call_to_cli_args,
)
from inference.classes import (
    CHAT_MODES,
    TOOL_USE_MODES,
    ChatMessage,
    InferenceRequest,
    LoadTextInferenceCall,
    LoadTextInferenceInit,
    ChatHistory,
    DEFAULT_CONTEXT_WINDOW,
)

# Minimum content length before checking for CLI turn marker (">").
# This prevents early termination on ">" characters that appear in the model's
# actual output (e.g., in code snippets, comparisons, or quoted text).
MIN_CONTENT_LENGTH_FOR_CLI_MARKER = 100

# Timeout in seconds for waiting on process termination during cleanup
PROCESS_TERMINATION_TIMEOUT = 5.0


# https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#common-options
class LLAMA_CPP:
    """Run a llama.cpp cli binary"""

    def __init__(
        self,
        model_path: str,  # Or, you can set the path to a pre-downloaded model file instead of model_url
        model_name: str,  # Friendly name
        model_id: str,  #  id of model in config
        response_mode: CHAT_MODES,  # CHAT_MODES
        model_url: str = None,  # Provide a url to download a model from
        mmproj_path: str = None,  # Path to mmproj file for vision capability
        raw_input: bool = None,  # user can send manually formatted messages
        func_calling: TOOL_USE_MODES = None,  # Function calling method (TOOL_USE_MODES)
        tool_schema_type: str = None,  # Determines which format of func definition should be fed to prompt (currently only applies to native)
        message_format: Optional[dict] = {},  # template converts messages to prompts
        verbose=False,
        debug=False,  # Show logs
        model_init_kwargs: LoadTextInferenceInit = None,  # kwargs to pass when loading the model
        generate_kwargs: LoadTextInferenceCall = None,  # kwargs to pass when generating text
    ):
        # Set args
        n_ctx = model_init_kwargs.n_ctx or DEFAULT_CONTEXT_WINDOW
        if n_ctx <= 0:
            n_ctx = DEFAULT_CONTEXT_WINDOW
        n_threads = model_init_kwargs.n_threads
        n_gpu_layers = model_init_kwargs.n_gpu_layers
        if model_init_kwargs.n_gpu_layers == -1:
            n_gpu_layers = 999  # offload all layers (diff models have diff max layers)

        init_kwargs = {
            "--n-gpu-layers": n_gpu_layers,
            "--no-mmap": not model_init_kwargs.use_mmap,
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
        self.tool_schema_type = tool_schema_type
        self.max_empty = 100
        self.chat_history = None
        self.process: Process = None
        self.process_type = ""
        self.task_logging = None
        self.abort_requested = False
        self.prompt_template = None  # structures the llm thoughts (thinking)
        self.message_format = message_format
        self.response_mode = response_mode
        self.func_calling = func_calling
        self.raw_input = raw_input or False
        self.model_url = model_url
        self.model_name = model_name or "chatbot"  # human friendly name for display
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.model_path = model_path  # text-models/your-model.gguf
        self.mmproj_path = mmproj_path  # mmproj file for vision (None if text-only)
        self.model_init_kwargs = init_kwargs
        self._generate_kwargs = (
            inference_call_to_cli_args(generate_kwargs) if generate_kwargs else None
        )
        deps_path = common.dep_path()
        BINARY_FOLDER_PATH = os.path.join(deps_path, "servers", "llama.cpp")
        # Platform-aware binary name - use llama-mtmd-cli for vision, llama-cli for text-only
        if mmproj_path and os.path.exists(mmproj_path):
            binary_name = (
                "llama-mtmd-cli.exe"
                if platform.system() == "Windows"
                else "llama-mtmd-cli"
            )
        else:
            binary_name = (
                "llama-cli.exe" if platform.system() == "Windows" else "llama-cli"
            )
        self.BINARY_PATH: str = os.path.join(BINARY_FOLDER_PATH, binary_name)

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
            "--seed": (
                settings.seed
                if settings.seed is not None
                else self.model_init_kwargs.get("--seed")
            ),
            "--n-predict": settings.max_tokens,
        }
        if stopwords:
            # Never use an empty string like [""] or empty array []
            kwargs.update({"--reverse-prompt": stopwords})
        # @TODO This is the grammar string passed from webui, but we should pass a filename that loads a grammar file using --grammar-file
        # Useful if we want only emojis or react code as output
        if grammar:
            kwargs.update({"--grammar": grammar})
        self._generate_kwargs = kwargs

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
    async def unload(self):
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
                # Wait for process to fully terminate to avoid zombie processes
                try:
                    await asyncio.wait_for(
                        self.process.wait(), timeout=PROCESS_TERMINATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    print(
                        f"{common.PRNT_LLAMA_LOG} Process did not terminate within "
                        f"{PROCESS_TERMINATION_TIMEOUT}s timeout",
                        flush=True,
                    )

        except ProcessLookupError as e:
            print(f"{common.PRNT_LLAMA_LOG} Could not find process to kill: {e}")
        except Exception as e:
            print(f"{common.PRNT_LLAMA_LOG} Error occurred: {e}")
        finally:
            self.process = None
            self.task_logging = None

    async def _drain_startup_banner(self, timeout: float = 120.0):
        """Drain the startup banner from stdout in -cnv mode.
        Reads and discards all output until the interactive prompt appears,
        indicating llama-cli is ready for user input."""
        buffer = ""
        try:
            while True:
                byte = await asyncio.wait_for(
                    self.process.stdout.read(1), timeout=timeout
                )
                if not byte:
                    # Process exited — read stderr for crash info
                    stderr_output = ""
                    if self.process and self.process.stderr:
                        try:
                            stderr_data = await asyncio.wait_for(
                                self.process.stderr.read(), timeout=2.0
                            )
                            stderr_output = stderr_data.decode(
                                "utf-8", errors="ignore"
                            ).strip()
                        except asyncio.TimeoutError:
                            pass
                    print(
                        f"{common.PRNT_LLAMA} Process exited during banner drain."
                        f"\nStdout captured:\n{buffer[-500:]}"
                        f"\nStderr:\n{stderr_output[-1000:]}",
                        flush=True,
                    )
                    raise Exception(
                        f"llama-cli process exited during startup. {stderr_output[-500:]}"
                    )
                char = byte.decode("utf-8", errors="ignore")
                buffer += char
                # In -cnv mode, llama-cli prints "> " when ready for input
                if buffer.endswith("> "):
                    print(
                        f"{common.PRNT_LLAMA} Banner drained, process ready.",
                        flush=True,
                    )
                    break
        except asyncio.TimeoutError:
            print(
                f"{common.PRNT_LLAMA} Timeout waiting for banner drain ({timeout}s)",
                flush=True,
            )

    @staticmethod
    def _strip_startup_banner(content: str) -> str:
        """Strip the llama.cpp startup banner from captured stdout.
        Newer versions (b8252+) print the banner to stdout instead of stderr."""
        banner_markers = ["modalities :", "available commands:"]
        lines = content.split("\n")
        last_banner_line = -1
        for i, line in enumerate(lines):
            stripped = line.strip().lower()
            for marker in banner_markers:
                if marker in stripped:
                    last_banner_line = i
        if last_banner_line >= 0:
            # Skip indented command lines and blank lines after the marker
            idx = last_banner_line + 1
            while idx < len(lines) and (
                lines[idx].startswith("  ") or lines[idx].strip() == ""
            ):
                idx += 1
            return "\n".join(lines[idx:]).strip()
        return content

    @staticmethod
    def _clean_response(content: str) -> str:
        """Clean model response of artifacts from llama.cpp CLI output."""
        import re

        # Log thinking blocks before stripping them
        for pattern in [
            r"\[Start thinking\](.*?)\[End thinking\]",
            r"<think>(.*?)</think>",
        ]:
            for match in re.finditer(pattern, content, flags=re.DOTALL):
                print(
                    f"{common.PRNT_LLAMA} Thinking: {match.group(1).strip()}",
                    flush=True,
                )

        # Strip thinking blocks: [Start thinking]...[End thinking] or <think>...</think>
        content = re.sub(
            r"\[Start thinking\].*?\[End thinking\]",
            "",
            content,
            flags=re.DOTALL,
        )
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # Strip CLI metrics line: [ Prompt: X t/s | Generation: Y t/s ]
        content = re.sub(r"\[\s*Prompt:.*?Generation:.*?\]", "", content)

        # Strip trailing CLI prompt ">"
        content = re.sub(r"\n\s*>\s*$", "", content)

        return content.strip()

    async def read_logs(self):
        async for line in self.process.stderr:
            # Print logs in real-time
            print(f"{common.PRNT_LLAMA_LOG}", line.decode("utf-8").strip())
            # @TODO Check debug logs of llama.cpp for events
            # ...

    # Load a previous chat conversation
    # @TODO Not implemented, needs chat_to_completions(chat_history) to convert conversation to string
    async def load_chat(self, chat_history: List[ChatMessage] | None):
        if not chat_history:
            return
        return

    # User can pause Ai and add more input, keeps conn open.
    # @TODO Yet to be implemented
    async def text_collab(self):
        self.abort_requested = False
        # Create arguments for starting server
        cmd_args = [
            self.BINARY_PATH,
            "--model",
            self.model_path,
            "--no-display-prompt",
            "--log-disable",
            "--no-context-shift",
            "--simple-io",
            "--multiline-input",  # dont have to type "/" to add a new line
            "--interactive",
            "--ignore-eos -n -1",  # infinite response
        ]
        return

    async def pause_text_chat(self):
        # Send command to llama-cli
        print(f"{common.PRNT_LLAMA} Pausing chat generation")
        # Halts the process and returns control to user
        os.kill(self.process.pid, signal.CTRL_C_EVENT)

    # Send multi-turn messages by role. Message does not require formatting. Does not use tools.
    # Cannot reload chat history from cache.
    # May be easier to re-implement chat using /completion and loading kv_cache each call.
    # @TODO Would like to use /text_completion to perform chat
    async def text_chat(
        self,
        prompt: str,
        request: Request,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
        constrain_json_output: Optional[dict] = None,
    ):
        prompt_file_path = None
        try:
            self.abort_requested = False

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
            # Add conditional args
            if constrain_json_output:
                # Constrain output using json schema
                cmd_args.append("--json-schema")
                cmd_args.append(json.dumps(constrain_json_output))
            # Sets system message when using "-cnv" mode.
            # Use temp file to avoid Windows command line length limit (8191 chars).
            # The user prompt is sent via stdin (not a CLI arg), so it is not
            # subject to the Windows char limit and does not need a temp file.
            if system_message:
                prompt_file_path = write_prompt_to_temp_file(system_message)
                cmd_args.append("--system-prompt-file")  # dedicated system prompt flag
                cmd_args.append(prompt_file_path)
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
                await self._run(cmd_args)
                # Read logs
                if self.debug:
                    self.task_logging = asyncio.create_task(self.read_logs())
                # Drain startup banner from stdout before sending prompt
                # In -cnv mode, llama-cli prints banner + "> " to stdout
                await self._drain_startup_banner()
            # Send command to llama-cli
            # In --multiline-input mode, lines without trailing "\" are submitted
            # immediately. Escape internal newlines so the entire prompt is sent
            # as a single message, with the final line unescaped to trigger submission.
            command = prompt.strip().replace("\n", "\\\n")
            print(f"{common.PRNT_LLAMA} Generating chat with command: {command}")
            self.process.stdin.write(command.encode("utf-8") + b"\n")
            await self.process.stdin.drain()
            # Text generation - generator takes ownership of temp file cleanup
            generator = self._text_generator(
                stream=stream,
                gen_type="chat",
                request=request,
                prompt_file_path=prompt_file_path,
            )
            prompt_file_path = None  # Generator owns cleanup now
            return generator
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled.", flush=True)
        except (ValueError, UnicodeEncodeError) as e:
            print(f"{common.PRNT_LLAMA} Error querying llama.cpp: {e}", flush=True)
            raise Exception(f"Failed to query llama.cpp: {e}")
        finally:
            cleanup_temp_file(prompt_file_path)

    # Predict the rest of the prompt. Can work with tools.
    # FYI, if we want to load prev conversation from cache and continue from there, most likely must use completion since -cnv mode makes it difficult to reload and manage chat history.
    async def text_completion(
        self,
        prompt: str,
        request: Request,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
        # tools for models trained for func calling
        native_tool_defs: Optional[str] = None,
        constrain_json_output: Optional[dict] = None,
    ):
        prompt_file_path = None
        try:
            self.abort_requested = False

            # If format type provided pass input unchanged, llama.cpp will handle it?
            formatted_prompt = prompt.strip()
            if not self.raw_input:
                # Format prompt with model template
                formatted_prompt = completion_to_prompt(
                    user_message=prompt,
                    system_message=system_message,
                    messageFormat=self.message_format,
                    native_tool_defs=native_tool_defs,
                )

            # Debug: log the formatted prompt written to temp file
            # print(
            #     f"[LLAMA] Formatted prompt for temp file:\n{formatted_prompt[:2000]}..."
            # )

            # Write prompt to temp file to avoid Windows command line length limit (8191 chars)
            # Using --file instead of --prompt prevents [WinError 206] for large prompts
            prompt_file_path = write_prompt_to_temp_file(formatted_prompt)

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
                "--file",
                prompt_file_path,
            ]
            # Add conditional args
            if constrain_json_output:
                # Constrain output using json schema
                cmd_args.append("--json-schema")
                cmd_args.append(json.dumps(constrain_json_output))
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
            await self._run(cmd_args)
            # Read logs
            if self.debug:
                self.task_logging = asyncio.create_task(self.read_logs())
            # Send command to llama-cli
            await self.process.stdin.drain()
            # Text generation - generator takes ownership of temp file cleanup
            generator = self._text_generator(
                stream=stream,
                gen_type="completion",
                request=request,
                prompt_file_path=prompt_file_path,
            )
            prompt_file_path = None  # Generator owns cleanup now
            return generator
        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Streaming task was cancelled", flush=True)
        except (ValueError, UnicodeEncodeError) as e:
            print(f"{common.PRNT_LLAMA} Error querying llama.cpp: {e}", flush=True)
            raise Exception(f"Failed to query llama.cpp: {e}")
        finally:
            cleanup_temp_file(prompt_file_path)

    # Start llm process
    async def _run(self, cmd_args):
        creation_kwargs = {}
        # Check each platform and assign flags as necessary
        if platform.system() == "Windows":
            # Removes the terminal window when process runs (Win only)
            creation_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        # @TODO Incorporate with model_init_kwargs
        self.process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            bufsize=0,
            **creation_kwargs,
        )

    async def _cleanup_vision_process(self):
        """Terminate the vision subprocess, wait for it to exit, and cancel logging task."""
        try:
            if self.task_logging:
                self.task_logging.cancel()
            if self.process:
                if self.process.returncode is None:
                    self.process.terminate()
                    await self.process.wait()
                self.process = None
        except ProcessLookupError:
            if self.debug:
                print(
                    f"{common.PRNT_LLAMA} Process already terminated (clean exit)",
                    flush=True,
                )
            self.process = None
        except Exception as e:
            print(
                f"{common.PRNT_LLAMA} Cleanup error (non-fatal): {e}",
                flush=True,
            )
            self.process = None

    async def _text_generator(
        self,
        stream: bool,
        gen_type: str,
        request: Request,
        prompt_file_path: Optional[str] = None,
    ):
        """Parse incoming tokens/text and stop generation when necessary.
        Buffers output to strip the startup banner and thinking blocks before
        streaming tokens to the client."""
        import re

        self.process_type = gen_type
        content = ""
        marker_num = 0
        was_aborted = False
        decoder = codecs.getincrementaldecoder("utf-8")()
        eos_llama_token = "[end of text]"  # Corresponds to special token (number 2) in LLaMa embedding
        has_gen_started = False
        # Buffering state: absorb banner + thinking blocks before streaming
        is_buffering = True
        buffer = ""
        # Patterns that mark the end of content we want to discard
        think_end_markers = ["[End thinking]", "</think>"]
        banner_end_markers = ["available commands:", "modalities :"]
        try:
            # Start of generation
            yield event_payload(FEEDING_PROMPT)
            while True:
                # Handle abort signal or non-existent process
                aborted = await request.is_disconnected() if request else False
                if aborted or not self.process or self.abort_requested:
                    print(f"{common.PRNT_LLAMA} Text generation aborted", flush=True)
                    was_aborted = True
                    self.abort_requested = False  # Reset for next request
                    break
                # Attempt to read bytes and convert to text
                try:
                    byte = await self.process.stdout.read(1)
                    # Bail on empty
                    if not byte:
                        break
                    if not has_gen_started:
                        has_gen_started = True
                        yield event_payload(GENERATING_TOKENS)
                    # Read and parse bytes into text incrementally, handles multi-byte decoding
                    byte_text = decoder.decode(byte)
                # Stop incomplete bytes from passing
                except (UnicodeEncodeError, UnicodeDecodeError) as e:
                    continue
                # Add text to accumulated content
                content += byte_text

                # --- Buffering phase: absorb banner and thinking blocks ---
                if is_buffering:
                    buffer += byte_text
                    # Check if we've passed a thinking end marker
                    found_think_end = any(
                        m in buffer for m in think_end_markers
                    )
                    if found_think_end:
                        # Log thinking content before discarding
                        for pattern in [
                            r"\[Start thinking\](.*?)\[End thinking\]",
                            r"<think>(.*?)</think>",
                        ]:
                            for match in re.finditer(
                                pattern, buffer, flags=re.DOTALL
                            ):
                                print(
                                    f"{common.PRNT_LLAMA} Thinking: {match.group(1).strip()}",
                                    flush=True,
                                )
                        # Strip everything up to and including the think end marker
                        for marker in think_end_markers:
                            idx = buffer.rfind(marker)
                            if idx >= 0:
                                buffer = buffer[idx + len(marker) :]
                                break
                        # Flush remaining buffer as streamable content
                        is_buffering = False
                        cleaned = buffer.strip()
                        if stream and cleaned:
                            for char in cleaned:
                                yield json.dumps(token_payload(char))
                        buffer = ""
                        continue
                    # Check if banner ended (no thinking block follows)
                    found_banner_end = any(
                        m in buffer.lower() for m in banner_end_markers
                    )
                    # After banner, look for the "> " ready prompt to know banner is done
                    if found_banner_end and buffer.rstrip().endswith(">"):
                        # Discard the entire banner
                        is_buffering = False
                        buffer = ""
                        continue
                    # Safety: if buffer grows large without finding markers, stop buffering
                    # (model may not have banner or thinking blocks)
                    if len(buffer) > 8000:
                        is_buffering = False
                        # No markers found — the buffer is likely real content
                        if stream:
                            for char in buffer:
                                yield json.dumps(token_payload(char))
                        buffer = ""
                    continue

                # --- Normal streaming phase ---
                # Bail if end of sequence token found
                if content.endswith(eos_llama_token):
                    break
                # Check CLI "turn" token — in chat mode, the banner "> "
                # is consumed by _drain_startup_banner or the buffer phase,
                # so the first ">" here is the end-of-turn marker.
                if byte_text == ">" and gen_type == "chat":
                    marker_num += 1
                    if marker_num >= 1:
                        break
                if gen_type == "chat" and (
                    content.endswith("\r\n>") or content.endswith("\n>")
                ):
                    break
                # Send tokens
                if stream:
                    payload = token_payload(byte_text)
                    yield json.dumps(payload)  # streaming format expects json
            # Cleanup
            if self.task_logging:
                self.task_logging.cancel()
            # Only terminate cli if Instruct mode
            if self.process and gen_type == "completion":
                self.process.terminate()
                self.process = None
            # Skip sending final response if aborted
            if was_aborted:
                return
            # Finally, send all tokens together
            content += decoder.decode(b"", final=True)
            content = content.rstrip(eos_llama_token).strip()
            # Strip startup banner that newer llama.cpp prints to stdout
            content = self._strip_startup_banner(content)
            # Clean response artifacts (thinking blocks, metrics, CLI prompt)
            content = self._clean_response(content)
            if not content:
                errMsg = "No response from model. Check available memory, try to lower amount of GPU Layers or offload to CPU only."
                print(f"{common.PRNT_LLAMA} {errMsg}")
                # Return error msg
                raise Exception(f"{errMsg}")
            payload = content_payload(content)
            if stream:
                yield json.dumps(payload)
            else:
                yield payload
        finally:
            cleanup_temp_file(prompt_file_path)

    # Vision inference - requires mmproj to be loaded
    async def vision_completion(
        self,
        prompt: str,
        image_paths: List[str],
        request: Optional[Request] = None,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
    ):
        """
        Generate text response based on image(s) and text prompt.
        Requires model to be loaded with mmproj (via /vision/load).
        """
        prompt_file_path = None
        try:
            self.abort_requested = False

            # Verify mmproj is available
            if not self.mmproj_path:
                raise Exception(
                    "Vision completion requires mmproj. Load model via /vision/load."
                )

            # Verify binary exists
            if not os.path.exists(self.BINARY_PATH):
                raise Exception(
                    f"llama-mtmd-cli binary not found at: {self.BINARY_PATH}"
                )

            # Verify model files exist
            if not os.path.exists(self.model_path):
                raise Exception(f"Model file not found at: {self.model_path}")
            if not os.path.exists(self.mmproj_path):
                raise Exception(f"mmproj file not found at: {self.mmproj_path}")

            # Verify image files exist
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    raise Exception(f"Image file not found at: {img_path}")

            # Build command arguments
            # Note: llama-mtmd-cli has different flags than llama-cli
            # It doesn't support --no-display-prompt or --simple-io
            cmd_args = [
                self.BINARY_PATH,
                "--model",
                self.model_path,
                "--mmproj",
                self.mmproj_path,
            ]

            # Add image paths
            for img_path in image_paths:
                cmd_args.extend(["--image", img_path])

            # Build prompt with system message if provided
            full_prompt = prompt.strip()
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt.strip()}"

            # Write prompt to temp file to avoid Windows command line length limit (8191 chars)
            # Using --file instead of --prompt prevents [WinError 206] for large prompts
            prompt_file_path = write_prompt_to_temp_file(full_prompt)

            cmd_args.extend(["--file", prompt_file_path])

            # Build vision-specific args explicitly (llama-mtmd-cli has limited options)
            vision_args = {
                # From model_init_kwargs
                "--n-gpu-layers": self.model_init_kwargs.get("--n-gpu-layers"),
                "--ctx-size": self.model_init_kwargs.get("--ctx-size"),
                "--batch-size": self.model_init_kwargs.get("--batch-size"),
                "--threads": self.model_init_kwargs.get("--threads"),
                "--seed": self.model_init_kwargs.get("--seed"),
                "--cache-type-k": self.model_init_kwargs.get("--cache-type-k"),
                "--cache-type-v": self.model_init_kwargs.get("--cache-type-v"),
            }
            # Apply overrides (temperature, max_tokens, etc.)
            if override_args:
                vision_args.update(override_args)

            # Remove None values
            vision_args = {k: v for k, v in vision_args.items() if v is not None}
            sanitized = sanitize_kwargs(kwargs=vision_args)
            cmd_args.extend(sanitized)

            # Start process
            print(
                f"{common.PRNT_LLAMA} Starting llama-mtmd-cli for vision...",
                flush=True,
            )
            print(
                f"{common.PRNT_LLAMA} Binary: {self.BINARY_PATH}",
                flush=True,
            )
            print(
                f"{common.PRNT_LLAMA} Model: {self.model_path}",
                flush=True,
            )
            print(
                f"{common.PRNT_LLAMA} mmproj: {self.mmproj_path}",
                flush=True,
            )
            print(
                f"{common.PRNT_LLAMA} Images: {image_paths}",
                flush=True,
            )
            if self.verbose:
                print(
                    f"{common.PRNT_LLAMA} Full command: {' '.join(str(arg) for arg in cmd_args)}",
                    flush=True,
                )
            await self._run(cmd_args)

            # Read logs
            if self.debug:
                self.task_logging = asyncio.create_task(self.read_logs())

            # Drain stdin
            await self.process.stdin.drain()

            # Generate response - generator takes ownership of temp file cleanup
            generator = self._vision_generator(
                stream=stream, request=request, prompt_file_path=prompt_file_path
            )
            prompt_file_path = None  # Generator owns cleanup now
            return generator

        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Vision task was cancelled", flush=True)
            # Match text_completion behavior - silent cancellation, don't re-raise
        except (ValueError, UnicodeEncodeError) as e:
            err_msg = str(e) if str(e) else f"{type(e).__name__} (no message)"
            print(
                f"{common.PRNT_LLAMA} Error in vision inference: {err_msg}", flush=True
            )
            raise Exception(f"Failed vision inference: {err_msg}")
        finally:
            cleanup_temp_file(prompt_file_path)

    async def _vision_generator(
        self,
        stream: bool,
        request: Optional[Request] = None,
        prompt_file_path: Optional[str] = None,
    ):
        """Parse incoming tokens and yield response for vision inference"""
        content = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        eos_token = "[end of text]"
        has_gen_started = False
        break_reason = "unknown"

        try:
            # Start of generation
            yield event_payload(FEEDING_PROMPT)

            while True:
                # Handle abort (only check if request context exists)
                aborted = await request.is_disconnected() if request else False
                if aborted or not self.process or self.abort_requested:
                    print(f"{common.PRNT_LLAMA} Vision generation aborted", flush=True)
                    break_reason = "aborted"
                    self.abort_requested = False  # Reset for next request
                    break

                try:
                    byte = await self.process.stdout.read(1)
                    if not byte:
                        break_reason = "eof"
                        break

                    if not has_gen_started:
                        has_gen_started = True
                        yield event_payload(GENERATING_TOKENS)

                    byte_text = decoder.decode(byte)
                except (UnicodeEncodeError, UnicodeDecodeError) as e:
                    print(
                        f"{common.PRNT_LLAMA} Decode error (skipping byte): {e}",
                        flush=True,
                    )
                    continue

                # Check for end of sequence
                if content.endswith(eos_token):
                    break_reason = "eos_token"
                    break

                # Check for CLI turn marker (only after we have some content)
                # This prevents breaking on ">" characters in the model's output
                if (
                    byte_text == ">"
                    and len(content) > MIN_CONTENT_LENGTH_FOR_CLI_MARKER
                ):
                    break_reason = "cli_marker"
                    break

                content += byte_text

                # Stream tokens
                if stream:
                    payload = token_payload(byte_text)
                    yield json.dumps(payload)

            # Terminate process
            await self._cleanup_vision_process()

            # Skip sending final response if aborted
            if break_reason == "aborted":
                return

            # Finalize content
            content += decoder.decode(b"", final=True)
            content = content.rstrip(eos_token).strip()

            if self.verbose:
                print(
                    f"{common.PRNT_LLAMA} Generation complete. Reason: {break_reason}, Content length: {len(content)}",
                    flush=True,
                )

            if not content:
                # Try to capture stderr for debugging
                stderr_output = ""
                if self.process and self.process.stderr:
                    try:
                        stderr_data = await asyncio.wait_for(
                            self.process.stderr.read(), timeout=1.0
                        )
                        stderr_output = stderr_data.decode(
                            "utf-8", errors="ignore"
                        ).strip()
                    except asyncio.TimeoutError:
                        pass

                if stderr_output:
                    print(f"{common.PRNT_LLAMA_LOG} stderr output:\n{stderr_output}")

                errMsg = (
                    "No response from vision model. Check available memory, "
                    "ensure mmproj file is valid, or try lowering GPU layers."
                )
                if stderr_output:
                    errMsg += f"\nBinary stderr: {stderr_output[:500]}"

                print(f"{common.PRNT_LLAMA} {errMsg}")
                raise Exception(errMsg)

            payload = content_payload(content)
            if stream:
                yield json.dumps(payload)
            else:
                yield payload
        finally:
            cleanup_temp_file(prompt_file_path)
