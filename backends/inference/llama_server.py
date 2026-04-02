"""
LlamaServer: Run inference via llama-server HTTP API.

Replaces the CLI-based LLAMA_CPP class with HTTP requests to a long-running
llama-server process. Follows the same pattern as vision/embedding_server.py.
"""

import os
import json
import base64
import asyncio
import subprocess
import platform
from asyncio.subprocess import Process
from fastapi import Request
from typing import List, Optional
import httpx
from core import common
from inference.helpers import (
    FEEDING_PROMPT,
    GENERATING_TOKENS,
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

LOG_PREFIX = f"{common.bcolors.OKCYAN}[LLAMA-SERVER]{common.bcolors.ENDC}"

# Timeout for server health check polling (seconds)
SERVER_READY_TIMEOUT = 120

# Timeout for waiting on process termination during cleanup
PROCESS_TERMINATION_TIMEOUT = 5.0

# CLI arg name -> HTTP API body parameter name
CLI_TO_API = {
    "--temp": "temperature",
    "--top-k": "top_k",
    "--top-p": "top_p",
    "--min-p": "min_p",
    "--repeat-penalty": "repeat_penalty",
    "--presence-penalty": "presence_penalty",
    "--frequency-penalty": "frequency_penalty",
    "--n-predict": "n_predict",
    "--mirostat-ent": "mirostat_tau",
    "--seed": "seed",
}


def _override_args_to_api(override_args: dict) -> dict:
    """Convert CLI-style override args (e.g. {'--temp': 0.5}) to API params."""
    api_params = {}
    for cli_key, value in override_args.items():
        if cli_key in CLI_TO_API:
            api_params[CLI_TO_API[cli_key]] = value
    return api_params


class LlamaServer:
    """Run inference via llama-server HTTP API (replaces LLAMA_CPP CLI)."""

    def __init__(
        self,
        model_path: str,
        model_name: str,
        model_id: str,
        response_mode: CHAT_MODES,
        model_url: str = None,
        mmproj_path: str = None,
        raw_input: bool = None,
        func_calling: TOOL_USE_MODES = None,
        tool_schema_type: str = None,
        message_format: Optional[dict] = None,
        verbose=False,
        debug=False,
        model_init_kwargs: LoadTextInferenceInit = None,
        generate_kwargs: LoadTextInferenceCall = None,
    ):
        # Build model init kwargs (used as CLI flags for server startup)
        n_ctx = model_init_kwargs.n_ctx or DEFAULT_CONTEXT_WINDOW
        if n_ctx <= 0:
            n_ctx = DEFAULT_CONTEXT_WINDOW
        n_threads = model_init_kwargs.n_threads
        n_gpu_layers = model_init_kwargs.n_gpu_layers
        if model_init_kwargs.n_gpu_layers == -1:
            n_gpu_layers = 999

        init_kwargs = {
            "--n-gpu-layers": n_gpu_layers,
            "--no-mmap": not model_init_kwargs.use_mmap,
            "--mlock": model_init_kwargs.use_mlock,
            "--seed": model_init_kwargs.seed,
            "--ctx-size": n_ctx,
            "--batch-size": model_init_kwargs.n_batch,
            "--no-kv-offload": not model_init_kwargs.offload_kqv,
            "--cache-type-k": model_init_kwargs.cache_type_k,
            "--cache-type-v": model_init_kwargs.cache_type_v,
        }
        if n_threads is not None:
            init_kwargs["--threads"] = n_threads
            init_kwargs["--threads-batch"] = n_threads

        # Instance vars (same as LLAMA_CPP for interface compat)
        self.tool_schema_type = tool_schema_type
        self.chat_history = None
        self.process: Process = None
        self.process_type = ""
        self.task_logging = None
        self._last_stderr_lines: list = []  # recent stderr for crash diagnostics
        self._abort_event: Optional[asyncio.Event] = None  # per-request abort signal
        self._active_client: Optional[httpx.AsyncClient] = None  # active streaming client
        self.prompt_template = None
        self.message_format = message_format or {}
        self.response_mode = response_mode
        self.func_calling = func_calling
        self.raw_input = raw_input or False
        self.model_url = model_url
        self.model_name = model_name or "chatbot"
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.model_init_kwargs = init_kwargs
        self._is_ready = False

        # Convert generate kwargs to both CLI-style and API-style dicts
        self._generate_kwargs = (
            inference_call_to_cli_args(generate_kwargs) if generate_kwargs else {}
        )
        self._api_generate_kwargs = (
            self._build_api_kwargs_from_call(generate_kwargs) if generate_kwargs else {}
        )

        # Find llama-server binary
        deps_path = common.dep_path()
        binary_folder = os.path.join(deps_path, "servers", "llama.cpp")
        binary_name = (
            "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        )
        self.BINARY_PATH = os.path.join(binary_folder, binary_name)

        # Pick an available port via shared allocator.
        # If anything after this point raises, release the port so it isn't leaked.
        self.port = common.allocate_server_port()
        try:
            self.base_url = f"http://127.0.0.1:{self.port}"

            # Persistent HTTP client for streaming requests (closed on unload)
            self._http_client: Optional[httpx.AsyncClient] = None
        except Exception:
            common.release_server_port(self.port)
            raise

    def _build_api_kwargs_from_call(self, call: LoadTextInferenceCall) -> dict:
        """Build HTTP API kwargs from a LoadTextInferenceCall model."""
        params = {
            "temperature": call.temperature,
            "top_k": call.top_k,
            "top_p": call.top_p,
            "min_p": call.min_p,
            "repeat_penalty": call.repeat_penalty,
            "presence_penalty": call.presence_penalty,
            "frequency_penalty": call.frequency_penalty,
            "n_predict": call.max_tokens,
            "mirostat_tau": call.mirostat_tau,
        }
        if call.stop:
            params["stop"] = [call.stop] if isinstance(call.stop, str) else call.stop
        return params

    # ──────────────────────────────────────────────
    # Per-request abort signaling
    # ──────────────────────────────────────────────

    def _new_abort_event(self) -> asyncio.Event:
        """Create a fresh abort event for a new inference request."""
        self._abort_event = asyncio.Event()
        return self._abort_event

    @property
    def abort_requested(self) -> bool:
        """Check if the current request has been aborted."""
        return self._abort_event is not None and self._abort_event.is_set()

    @abort_requested.setter
    def abort_requested(self, value: bool):
        """Set or clear the abort signal. Setting True signals the current generator to stop."""
        if value:
            if self._abort_event is not None:
                self._abort_event.set()
            # Force-close active HTTP client to unblock hanging streams
            if self._active_client and not self._active_client.is_closed:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._active_client.aclose())
                except RuntimeError:
                    pass  # No running loop — generator cleanup will handle it
        else:
            # Reset is handled by _new_abort_event() at the start of each request
            pass

    # Getter - returns CLI-style dict for backward compat (used by get_text_model endpoint)
    @property
    def generate_kwargs(self) -> dict:
        return self._generate_kwargs

    # Setter - translates InferenceRequest to both CLI and API param dicts
    @generate_kwargs.setter
    def generate_kwargs(self, settings: InferenceRequest):
        self.prompt_template = settings.promptTemplate
        stopwords = settings.stop
        grammar = settings.grammar

        # CLI-style dict (for backward compat with get_text_model endpoint)
        self._generate_kwargs = {
            "--mirostat-ent": settings.mirostat_tau,
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
            self._generate_kwargs["--reverse-prompt"] = stopwords

        # API-style dict (for HTTP requests)
        self._api_generate_kwargs = {
            "temperature": settings.temperature,
            "top_k": settings.top_k,
            "top_p": settings.top_p,
            "min_p": settings.min_p,
            "repeat_penalty": settings.repeat_penalty,
            "presence_penalty": settings.presence_penalty,
            "frequency_penalty": settings.frequency_penalty,
            "n_predict": settings.max_tokens,
            "mirostat_tau": settings.mirostat_tau,
            "seed": (
                settings.seed
                if settings.seed is not None
                else self.model_init_kwargs.get("--seed")
            ),
        }
        if stopwords:
            self._api_generate_kwargs["stop"] = (
                [stopwords] if isinstance(stopwords, str) else stopwords
            )
        if grammar:
            self._api_generate_kwargs["grammar"] = grammar

    # ──────────────────────────────────────────────
    # Server lifecycle
    # ──────────────────────────────────────────────

    async def start_server(self) -> bool:
        """Start the llama-server process and wait for it to be ready."""
        if self._is_ready and self.process and self.process.returncode is None:
            return True

        # Validate binary
        if not os.path.exists(self.BINARY_PATH):
            raise FileNotFoundError(
                f"llama-server binary not found at {self.BINARY_PATH}"
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

        # Build command
        cmd = [
            self.BINARY_PATH,
            "-m", self.model_path,
            "--port", str(self.port),
            "--jinja",  # enable jinja chat templates
        ]

        # Add mmproj for vision models
        if self.mmproj_path and os.path.exists(self.mmproj_path):
            cmd.extend(["--mmproj", self.mmproj_path])

        # Add model init kwargs as CLI flags
        sanitized = sanitize_kwargs(kwargs=self.model_init_kwargs)
        cmd.extend(sanitized)

        cmd_str = " ".join(str(a) for a in cmd)
        print(f"{LOG_PREFIX} Starting llama-server on port {self.port}", flush=True)
        print(f"{LOG_PREFIX} Command: {cmd_str}", flush=True)

        # Write to a log file for debugging packaged builds (no console)
        log_path = os.path.join(common.app_path(), "llama-server.log")
        self._log_path = log_path

        # Spawn process
        creation_kwargs = {}
        if platform.system() == "Windows":
            creation_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        # Set cwd to binary directory so DLLs next to the exe are found
        binary_dir = os.path.dirname(self.BINARY_PATH)

        try:
            self._last_stderr_lines = []
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n--- start_server ---\n")
                f.write(f"Command: {cmd_str}\n")
                f.write(f"Binary dir (cwd): {binary_dir}\n")
                f.write(f"Model path exists: {os.path.exists(self.model_path)}\n")

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd=binary_dir,
                **creation_kwargs,
            )

            # Always drain stderr to prevent pipe buffer deadlock on Windows.
            # llama-server logs to stderr; if the buffer fills (~4KB) the
            # process blocks on write, which stalls HTTP responses.
            self.task_logging = asyncio.create_task(self._read_logs())

            # Wait for /health to respond
            self._is_ready = await self._wait_for_ready(timeout=SERVER_READY_TIMEOUT)

            if self._is_ready:
                print(f"{LOG_PREFIX} Server is ready on port {self.port}", flush=True)
                return True
            else:
                print(f"{LOG_PREFIX} Server failed to become ready", flush=True)
                await self.unload()
                return False

        except Exception as e:
            print(f"{LOG_PREFIX} Failed to start server: {e}", flush=True)
            await self.unload()
            raise

    async def _wait_for_ready(self, timeout: int = SERVER_READY_TIMEOUT) -> bool:
        """Poll /health until the server is ready."""
        health_url = f"{self.base_url}/health"
        start_time = asyncio.get_running_loop().time()

        async with httpx.AsyncClient() as client:
            while (asyncio.get_running_loop().time() - start_time) < timeout:
                try:
                    response = await client.get(health_url, timeout=2.0)
                    if response.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                # Check if process died
                if self.process and self.process.returncode is not None:
                    # Give _read_logs a moment to drain remaining output
                    await asyncio.sleep(0.2)
                    stderr_tail = "\n".join(self._last_stderr_lines)
                    msg = (
                        f"Server process died (exit code "
                        f"{self.process.returncode}):\n{stderr_tail}"
                    )
                    print(f"{LOG_PREFIX} {msg}", flush=True)
                    # Also write to log file for packaged builds
                    if hasattr(self, "_log_path"):
                        with open(self._log_path, "a", encoding="utf-8") as f:
                            f.write(f"{msg}\n")
                    return False

                await asyncio.sleep(1)

        return False

    async def _read_logs(self):
        """Drain server stderr to prevent pipe buffer deadlock on Windows.
        Always consumes output; only prints when debug is enabled.
        Keeps the last 20 lines for crash diagnostics."""
        try:
            async for line in self.process.stderr:
                text = line.decode("utf-8", errors="ignore").strip()
                if text:
                    self._last_stderr_lines.append(text)
                    # Keep only the last 20 lines to limit memory use
                    if len(self._last_stderr_lines) > 20:
                        self._last_stderr_lines.pop(0)
                if self.debug:
                    print(f"{LOG_PREFIX} {text}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"{LOG_PREFIX} Log reader error: {e}", flush=True)

    async def _ensure_server(self):
        """Start the server if not already running. Warns if restarting after a crash."""
        if self._is_ready and self.process and self.process.returncode is None:
            return
        if self.process and self.process.returncode is not None:
            print(
                f"{LOG_PREFIX} WARNING: Server process crashed (exit code "
                f"{self.process.returncode}). Restarting — KV cache and context "
                f"have been lost.",
                flush=True,
            )
        success = await self.start_server()
        if not success:
            raise RuntimeError("Failed to start llama-server")

    def _get_http_client(self) -> httpx.AsyncClient:
        """Return the persistent async HTTP client, creating one if needed."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient()
        return self._http_client

    async def unload(self):
        """Stop the server process and release resources."""
        self._is_ready = False
        try:
            # Close persistent HTTP client
            if self._http_client and not self._http_client.is_closed:
                await self._http_client.aclose()
            self._http_client = None
            if self.task_logging:
                self.task_logging.cancel()
            if self.process:
                print(f"{LOG_PREFIX} Stopping llama-server on port {self.port}", flush=True)
                try:
                    self.process.terminate()
                    await asyncio.wait_for(
                        self.process.wait(), timeout=PROCESS_TERMINATION_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    print(f"{LOG_PREFIX} Force killing server", flush=True)
                    self.process.kill()
                except ProcessLookupError:
                    pass
        except Exception as e:
            print(f"{LOG_PREFIX} Error stopping server: {e}", flush=True)
        finally:
            self.process = None
            self.task_logging = None
            common.release_server_port(self.port)

    # ──────────────────────────────────────────────
    # Chat history (same stubs as LLAMA_CPP)
    # ──────────────────────────────────────────────

    async def load_cached_chat(self, chat_history: Optional[ChatHistory] = None):
        if not chat_history:
            return
        self.chat_history = chat_history

    async def load_chat(self, chat_history: List[ChatMessage] | None):
        if not chat_history:
            return

    async def cancel_server_generation(self):
        """Cancel active generation on the server via POST /slots/0?action=erase.

        This tells llama-server to stop generating immediately rather than
        buffering remaining tokens. Uses slot 0 because this server is
        configured with a single slot (the default). If multi-slot support
        is added later, this will need to target the correct slot ID.
        """
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{self.base_url}/slots/0?action=erase", timeout=3.0
                )
                print(f"{LOG_PREFIX} Server-side generation cancelled", flush=True)
        except Exception as e:
            print(f"{LOG_PREFIX} Could not cancel server generation: {e}", flush=True)

    async def pause_text_chat(self):
        """No-op: server-based chat doesn't use stdin signals."""
        print(f"{LOG_PREFIX} Pause requested (aborting current stream)", flush=True)
        self.abort_requested = True

    # ──────────────────────────────────────────────
    # Text completion (INSTRUCT mode) - POST /completion
    # ──────────────────────────────────────────────

    async def text_completion(
        self,
        prompt: str,
        request: Request,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
        native_tool_defs: Optional[str] = None,
        constrain_json_output: Optional[dict] = None,
    ):
        try:
            abort_event = self._new_abort_event()
            await self._ensure_server()

            # Format prompt (same logic as LLAMA_CPP)
            formatted_prompt = prompt.strip()
            if not self.raw_input:
                formatted_prompt = completion_to_prompt(
                    user_message=prompt,
                    system_message=system_message,
                    messageFormat=self.message_format,
                    native_tool_defs=native_tool_defs,
                )

            # Build request body
            body = {
                "prompt": formatted_prompt,
                "stream": True,  # always stream from server, we aggregate if needed
                **self._api_generate_kwargs,
            }

            # Constrained output
            if constrain_json_output:
                body["json_schema"] = constrain_json_output

            # Apply overrides
            if override_args:
                body.update(_override_args_to_api(override_args))

            print(f"{LOG_PREFIX} Generating completion", flush=True)
            self.process_type = "completion"

            generator = self._completion_generator(
                body=body, stream=stream, request=request, abort_event=abort_event
            )
            return generator

        except asyncio.CancelledError:
            print(f"{LOG_PREFIX} Completion task was cancelled", flush=True)
            raise
        except Exception as e:
            print(f"{LOG_PREFIX} Error in text_completion: {e}", flush=True)
            raise Exception(f"Failed to query llama-server: {e}")

    async def _completion_generator(
        self, body: dict, stream: bool, request: Request,
        abort_event: asyncio.Event = None,
    ):
        """Stream tokens from POST /completion and yield SSE events."""
        content = ""
        has_gen_started = False
        was_aborted = False
        url = f"{self.base_url}/completion"

        client = self._get_http_client()
        self._active_client = client
        try:
            async with client.stream("POST", url, json=body, timeout=None) as response:
                response.raise_for_status()

                # Connection established and valid — safe to signal prompt feeding
                yield event_payload(FEEDING_PROMPT)

                async for line in response.aiter_lines():
                    # Check abort (per-request event, not shared boolean)
                    aborted = await request.is_disconnected() if request else False
                    if aborted or (abort_event and abort_event.is_set()):
                        print(f"{LOG_PREFIX} Generation aborted", flush=True)
                        was_aborted = True
                        break

                    # Parse SSE line: "data: {...}"
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    token_text = data.get("content", "")
                    if not token_text:
                        continue

                    if not has_gen_started:
                        has_gen_started = True
                        yield event_payload(GENERATING_TOKENS)

                    content += token_text

                    if stream:
                        yield json.dumps(token_payload(token_text))

                    # Check if generation is done
                    if data.get("stop"):
                        break

            # Skip final if aborted
            if was_aborted:
                return

            content = content.strip()
            if not content:
                errMsg = (
                    "No response from model. Check available memory, "
                    "try to lower amount of GPU Layers or offload to CPU only."
                )
                print(f"{LOG_PREFIX} {errMsg}")
                raise Exception(errMsg)

            payload = content_payload(content)
            if stream:
                yield json.dumps(payload)
            else:
                yield payload

        except httpx.HTTPStatusError as e:
            print(f"{LOG_PREFIX} HTTP error: {e.response.status_code}", flush=True)
            raise Exception(f"Server returned error: {e.response.status_code}")
        except (httpx.ReadError, httpx.CloseError, httpx.RemoteProtocolError):
            # Stream interrupted by abort handler closing the client
            if abort_event and abort_event.is_set():
                return
            raise
        finally:
            self._active_client = None

    # ──────────────────────────────────────────────
    # Text chat (CHAT mode) - POST /v1/chat/completions
    # ──────────────────────────────────────────────

    async def text_chat(
        self,
        prompt: str,
        request: Request,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
        constrain_json_output: Optional[dict] = None,
    ):
        try:
            abort_event = self._new_abort_event()
            await self._ensure_server()

            # Build messages array
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": prompt.strip()})

            # Build request body
            body = {
                "messages": messages,
                "stream": True,
                **self._api_generate_kwargs,
            }

            # Constrained output
            if constrain_json_output:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": constrain_json_output,
                }

            # Apply overrides
            if override_args:
                body.update(_override_args_to_api(override_args))

            print(f"{LOG_PREFIX} Generating chat response", flush=True)
            self.process_type = "chat"

            generator = self._chat_generator(
                body=body, stream=stream, request=request, abort_event=abort_event
            )
            return generator

        except asyncio.CancelledError:
            print(f"{LOG_PREFIX} Chat task was cancelled", flush=True)
            raise
        except Exception as e:
            print(f"{LOG_PREFIX} Error in text_chat: {e}", flush=True)
            raise Exception(f"Failed to query llama-server: {e}")

    async def _chat_generator(
        self, body: dict, stream: bool, request: Request,
        abort_event: asyncio.Event = None,
    ):
        """Stream tokens from POST /v1/chat/completions and yield SSE events."""
        content = ""
        has_gen_started = False
        was_aborted = False
        url = f"{self.base_url}/v1/chat/completions"

        client = self._get_http_client()
        self._active_client = client
        try:
            async with client.stream("POST", url, json=body, timeout=None) as response:
                response.raise_for_status()

                # Connection established and valid — safe to signal prompt feeding
                yield event_payload(FEEDING_PROMPT)

                async for line in response.aiter_lines():
                    # Check abort (per-request event, not shared boolean)
                    aborted = await request.is_disconnected() if request else False
                    if aborted or (abort_event and abort_event.is_set()):
                        print(f"{LOG_PREFIX} Chat generation aborted", flush=True)
                        was_aborted = True
                        break

                    # Parse SSE: "data: {...}"
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # OpenAI format: choices[0].delta.content
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    token_text = delta.get("content", "")
                    finish_reason = choices[0].get("finish_reason")

                    if token_text:
                        if not has_gen_started:
                            has_gen_started = True
                            yield event_payload(GENERATING_TOKENS)

                        content += token_text

                        if stream:
                            yield json.dumps(token_payload(token_text))

                    if finish_reason:
                        break

            # Skip final if aborted
            if was_aborted:
                return

            content = content.strip()
            if not content:
                errMsg = (
                    "No response from model. Check available memory, "
                    "try to lower amount of GPU Layers or offload to CPU only."
                )
                print(f"{LOG_PREFIX} {errMsg}")
                raise Exception(errMsg)

            payload = content_payload(content)
            if stream:
                yield json.dumps(payload)
            else:
                yield payload

        except httpx.HTTPStatusError as e:
            print(f"{LOG_PREFIX} HTTP error: {e.response.status_code}", flush=True)
            raise Exception(f"Server returned error: {e.response.status_code}")
        except (httpx.ReadError, httpx.CloseError, httpx.RemoteProtocolError):
            # Stream interrupted by abort handler closing the client
            if abort_event and abort_event.is_set():
                return
            raise
        finally:
            self._active_client = None

    # ──────────────────────────────────────────────
    # Vision completion - POST /v1/chat/completions with images
    # ──────────────────────────────────────────────

    async def vision_completion(
        self,
        prompt: str,
        image_paths: List[str],
        request: Optional[Request] = None,
        system_message: Optional[str] = None,
        stream: bool = False,
        override_args: Optional[dict] = None,
    ):
        try:
            abort_event = self._new_abort_event()
            await self._ensure_server()

            # Verify mmproj
            if not self.mmproj_path:
                raise Exception(
                    "Vision completion requires mmproj. Load model via /vision/load."
                )

            # Verify image files exist
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    raise Exception(f"Image file not found at: {img_path}")

            # Build multimodal content
            content_parts = []
            # Add images
            for img_path in image_paths:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")

                # Detect mime type
                ext = os.path.splitext(img_path)[1].lower()
                mime_types = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                mime = mime_types.get(ext, "image/png")

                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_data}"},
                })

            # Add text prompt
            content_parts.append({"type": "text", "text": prompt.strip()})

            # Build messages
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": content_parts})

            # Build request body
            body = {
                "messages": messages,
                "stream": True,
            }

            # Build vision-specific sampling params
            vision_api_args = {
                "temperature": self._api_generate_kwargs.get("temperature"),
                "n_predict": self._api_generate_kwargs.get("n_predict"),
                "top_k": self._api_generate_kwargs.get("top_k"),
                "top_p": self._api_generate_kwargs.get("top_p"),
                "min_p": self._api_generate_kwargs.get("min_p"),
                "repeat_penalty": self._api_generate_kwargs.get("repeat_penalty"),
                "seed": self._api_generate_kwargs.get("seed"),
            }
            # Remove None values
            vision_api_args = {k: v for k, v in vision_api_args.items() if v is not None}
            body.update(vision_api_args)

            # Apply overrides
            if override_args:
                body.update(_override_args_to_api(override_args))

            print(f"{LOG_PREFIX} Starting vision inference", flush=True)
            print(f"{LOG_PREFIX} Images: {image_paths}", flush=True)
            self.process_type = "completion"

            # Reuse the chat generator (same SSE format from /v1/chat/completions)
            generator = self._chat_generator(
                body=body, stream=stream, request=request, abort_event=abort_event
            )
            return generator

        except asyncio.CancelledError:
            print(f"{LOG_PREFIX} Vision task was cancelled", flush=True)
            raise
        except (ValueError, UnicodeEncodeError) as e:
            err_msg = str(e) if str(e) else f"{type(e).__name__} (no message)"
            print(f"{LOG_PREFIX} Error in vision inference: {err_msg}", flush=True)
            raise Exception(f"Failed vision inference: {err_msg}")
