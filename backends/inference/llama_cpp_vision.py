import os
import json
import codecs
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
    event_payload,
    token_payload,
    content_payload,
    sanitize_kwargs,
)
from inference.classes import (
    LoadTextInferenceCall,
    LoadTextInferenceInit,
    DEFAULT_CONTEXT_WINDOW,
)


# @TODO Wondering if this should be merged with llama_cpp ?
class LLAMA_CPP_VISION:
    """Run llama-mtmd-cli binary for multimodal (vision) inference"""

    def __init__(
        self,
        model_path: str,  # Path to main GGUF model file
        mmproj_path: str,  # Path to mmproj (multimodal projector) file - REQUIRED
        model_name: str,  # Friendly name
        model_id: str,  # Model config ID
        verbose: bool = False,
        debug: bool = False,
        model_init_kwargs: LoadTextInferenceInit = None,
        generate_kwargs: LoadTextInferenceCall = None,
    ):
        # Set args
        n_ctx = model_init_kwargs.n_ctx or DEFAULT_CONTEXT_WINDOW
        if n_ctx <= 0:
            n_ctx = DEFAULT_CONTEXT_WINDOW
        n_threads = model_init_kwargs.n_threads
        n_gpu_layers = model_init_kwargs.n_gpu_layers
        if model_init_kwargs.n_gpu_layers == -1:
            n_gpu_layers = 999  # offload all layers

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

        # Assign vars
        self.max_empty = 100
        self.process: Process = None
        self.task_logging = None
        self.abort_requested = False
        self.model_name = model_name or "vision-model"
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.model_init_kwargs = init_kwargs
        self._generate_kwargs = generate_kwargs

        # Binary path
        deps_path = common.dep_path()
        BINARY_FOLDER_PATH = os.path.join(deps_path, "servers", "llama.cpp")
        binary_name = (
            "llama-mtmd-cli.exe" if platform.system() == "Windows" else "llama-mtmd-cli"
        )
        self.BINARY_PATH: str = os.path.join(BINARY_FOLDER_PATH, binary_name)

    @property
    def generate_kwargs(self) -> dict:
        return self._generate_kwargs

    @generate_kwargs.setter
    def generate_kwargs(self, settings):
        """Translate settings to command line args"""
        kwargs = {
            "--temp": settings.temperature,
            "--top-k": settings.top_k,
            "--top-p": settings.top_p,
            "--min-p": settings.min_p,
            "--repeat-penalty": settings.repeat_penalty,
            "--ctx-size": self.model_init_kwargs.get("--ctx-size"),
            "--seed": self.model_init_kwargs.get("--seed"),
            "--n-predict": settings.max_tokens,
        }
        self._generate_kwargs = kwargs

    def unload(self):
        """Shutdown vision inference instance"""
        try:
            if self.task_logging:
                self.task_logging.cancel()
            if self.process:
                print(
                    f"{common.PRNT_LLAMA} Shutting down llama-mtmd-cli process.",
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
        """Read and print stderr logs"""
        async for line in self.process.stderr:
            print(f"{common.PRNT_LLAMA_LOG}", line.decode("utf-8").strip())

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
        Uses llama-mtmd-cli binary for multimodal inference.
        """
        try:
            self.abort_requested = False

            # Build command arguments
            cmd_args = [
                self.BINARY_PATH,
                "--model",
                self.model_path,
                "--mmproj",
                self.mmproj_path,
                "--no-display-prompt",
                "--simple-io",
            ]

            # Add image paths
            for img_path in image_paths:
                cmd_args.extend(["--image", img_path])

            # Build prompt with system message if provided
            full_prompt = prompt.strip()
            if system_message:
                full_prompt = f"{system_message}\n\n{prompt.strip()}"

            cmd_args.extend(["--prompt", full_prompt])

            # Merge and sanitize args
            merged_args = self.model_init_kwargs.copy()
            if self._generate_kwargs:
                merged_args.update(self._generate_kwargs)
            if override_args:
                merged_args.update(override_args)
            sanitized = sanitize_kwargs(kwargs=merged_args)
            cmd_args.extend(sanitized)

            # Start process
            print(
                f"{common.PRNT_LLAMA} Starting llama-mtmd-cli for vision...\n"
                f"With command: {cmd_args}"
            )
            await self._run(cmd_args)

            # Read logs
            if self.debug:
                self.task_logging = asyncio.create_task(self.read_logs())

            # Drain stdin
            await self.process.stdin.drain()

            # Generate response
            return self._vision_generator(stream=stream, request=request)

        except asyncio.CancelledError:
            print(f"{common.PRNT_LLAMA} Vision task was cancelled", flush=True)
        except (ValueError, UnicodeEncodeError, Exception) as e:
            print(f"{common.PRNT_LLAMA} Error in vision inference: {e}", flush=True)
            raise Exception(f"Failed vision inference: {e}")

    async def _run(self, cmd_args):
        """Start the llama-mtmd-cli process"""
        creation_kwargs = {}
        if platform.system() == "Windows":
            creation_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        self.process = await asyncio.create_subprocess_exec(
            *cmd_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            bufsize=0,
            **creation_kwargs,
        )

    async def _vision_generator(
        self,
        stream: bool,
        request: Optional[Request] = None,
    ):
        """Parse incoming tokens and yield response"""
        content = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        eos_token = "[end of text]"
        has_gen_started = False

        # Start of generation
        yield event_payload(FEEDING_PROMPT)

        while True:
            # Handle abort (only check if request context exists)
            aborted = await request.is_disconnected() if request else False
            if aborted or not self.process or self.abort_requested:
                print(f"{common.PRNT_LLAMA} Vision generation aborted", flush=True)
                break

            try:
                byte = await self.process.stdout.read(1)
                if not byte:
                    break

                if not has_gen_started:
                    has_gen_started = True
                    yield event_payload(GENERATING_TOKENS)

                byte_text = decoder.decode(byte)
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue

            # Check for end of sequence
            if content.endswith(eos_token):
                break

            # Check for CLI turn marker
            if byte_text == ">":
                break

            content += byte_text

            # Stream tokens
            if stream:
                payload = token_payload(byte_text)
                yield json.dumps(payload)

        # Finalize content
        content += decoder.decode(b"", final=True)
        content = content.rstrip(eos_token).strip()

        if not content:
            errMsg = (
                "No response from vision model. Check available memory, "
                "ensure mmproj file is valid, or try lowering GPU layers."
            )
            print(f"{common.PRNT_LLAMA} {errMsg}")
            if self.task_logging:
                self.task_logging.cancel()
            if self.process:
                self.process.terminate()
                self.process = None
            raise Exception(errMsg)

        payload = content_payload(content)
        if stream:
            yield json.dumps(payload)
        else:
            yield payload

        # Cleanup
        if self.task_logging:
            self.task_logging.cancel()
        if self.process:
            self.process.terminate()
            self.process = None
