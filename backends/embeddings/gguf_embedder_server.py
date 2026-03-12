"""
GGUFEmbedderServer: Handle GGUF embedding models via llama-server HTTP API.

Replaces the CLI-based GGUFEmbedder that spawned llama-embedding as a one-shot subprocess.
Follows the same pattern as vision/embedding_server.py.
"""

import os
import time
import platform
import subprocess
from typing import List
import httpx
from core import common
from core.classes import FastAPIApp

LOG_PREFIX = "[GGUF-EMBEDDER-SERVER]"

# Port range for the text embedding server (non-overlapping with LlamaServer 8082-8085)
DEFAULT_PORT = 8086
PORT_RANGE_START = 8086
PORT_RANGE_END = 8090

# Timeout for server readiness
SERVER_READY_TIMEOUT = 60


def _find_available_port(start: int = PORT_RANGE_START, end: int = PORT_RANGE_END) -> int:
    """Find an available port in the given range."""
    for port in range(start, end + 1):
        if common.check_open_port(port) != 0:
            return port
    raise RuntimeError(f"No available port found in range {start}-{end}")


class GGUFEmbedderServer:
    """Handle GGUF embedding models using llama-server with --embedding flag."""

    def __init__(
        self,
        app: FastAPIApp,
        model_path: str = None,
        embed_model: str = None,
    ):
        self.app = app
        self.model_name = embed_model or "GGUF Embedding Model"
        self.model_path = model_path
        self.process: subprocess.Popen = None
        self._is_ready = False

        # Find llama-server binary
        deps_path = common.dep_path()
        binary_name = (
            "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        )
        self.binary_path = os.path.join(deps_path, "servers", "llama.cpp", binary_name)

        # Verify binary exists
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(
                f"llama-server binary not found at {self.binary_path}. "
                "Please restart the server to download required binaries."
            )

        # Pick port
        self.port = _find_available_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        print(f"{LOG_PREFIX} Initialized with model: {self.model_name}", flush=True)
        print(f"{LOG_PREFIX} Binary path: {self.binary_path}", flush=True)
        print(f"{LOG_PREFIX} Model path: {self.model_path}", flush=True)

    def _start_server(self) -> bool:
        """Start llama-server with --embedding flag (synchronous).

        Uses subprocess.Popen so this can be called safely from background
        threads without needing an asyncio event loop.
        """
        if self._is_ready and self.process and self.process.poll() is None:
            return True

        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the GGUF embedding model is downloaded."
            )

        cmd = [
            self.binary_path,
            "-m", self.model_path,
            "--embedding",
            "--pooling", "mean",
            "--port", str(self.port),
            "-ngl", "99",
        ]

        print(f"{LOG_PREFIX} Starting embedding server on port {self.port}", flush=True)

        creation_kwargs = {}
        if platform.system() == "Windows":
            creation_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                **creation_kwargs,
            )

            self._is_ready = self._wait_for_ready(timeout=SERVER_READY_TIMEOUT)

            if self._is_ready:
                print(f"{LOG_PREFIX} Embedding server is ready", flush=True)
                return True
            else:
                print(f"{LOG_PREFIX} Server failed to become ready", flush=True)
                self._stop_process()
                return False

        except Exception as e:
            print(f"{LOG_PREFIX} Failed to start server: {e}", flush=True)
            self._stop_process()
            raise

    def _wait_for_ready(self, timeout: int = SERVER_READY_TIMEOUT) -> bool:
        """Poll /health synchronously until the server is ready."""
        health_url = f"{self.base_url}/health"
        start_time = time.monotonic()

        with httpx.Client() as client:
            while (time.monotonic() - start_time) < timeout:
                try:
                    response = client.get(health_url, timeout=2.0)
                    if response.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                if self.process and self.process.poll() is not None:
                    stderr = self.process.stderr.read() if self.process.stderr else b""
                    print(
                        f"{LOG_PREFIX} Server process died: {stderr.decode(errors='ignore')}",
                        flush=True,
                    )
                    return False

                time.sleep(1)

        return False

    def _stop_process(self):
        """Synchronously stop the server process."""
        if self.process:
            print(f"{LOG_PREFIX} Stopping embedding server", flush=True)
            try:
                self.process.terminate()
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                print(f"{LOG_PREFIX} Force killing server", flush=True)
                self.process.kill()
            except Exception as e:
                print(f"{LOG_PREFIX} Error stopping server: {e}", flush=True)
            finally:
                self.process = None

    async def stop(self):
        """Stop the embedding server (async interface for shutdown path)."""
        self._is_ready = False
        self._stop_process()

    def embed_text(self, text: str) -> List[float]:
        """
        Create vector embeddings from text using llama-server /embeddings endpoint.

        Uses synchronous HTTP calls because this method is called from
        background threads (via BackgroundTasks).
        """
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please ensure the GGUF embedding model is downloaded."
            )

        # Start server if needed (fully synchronous — no event loop required)
        if not self._is_ready:
            result = self._start_server()
            if not result:
                raise RuntimeError("Failed to start embedding server")

        print(
            f"{LOG_PREFIX} Embedding text of length {len(text)} characters",
            flush=True,
        )

        url = f"{self.base_url}/embeddings"
        payload = {"content": text}

        with httpx.Client() as client:
            try:
                response = client.post(url, json=payload, timeout=60.0)
                response.raise_for_status()

                data = response.json()

                # Parse response -- handle different formats
                raw_embedding = None
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and "embedding" in data[0]:
                        raw_embedding = data[0]["embedding"]
                    elif isinstance(data[0], list):
                        raw_embedding = data[0]
                elif isinstance(data, dict):
                    if "embedding" in data:
                        raw_embedding = data["embedding"]
                    elif "data" in data and isinstance(data["data"], list):
                        if len(data["data"]) > 0 and "embedding" in data["data"][0]:
                            raw_embedding = data["data"][0]["embedding"]

                if raw_embedding is not None:
                    print(
                        f"{LOG_PREFIX} Generated embedding with {len(raw_embedding)} dimensions",
                        flush=True,
                    )
                    return raw_embedding

                raise ValueError(f"Unexpected response format: {data}")

            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"Embedding server returned error: {e.response.status_code} - {e.response.text}"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate embedding: {e}")

    def get_text_embedding(self, text: str) -> List[float]:
        """Alias for embed_text for compatibility with HuggingFaceEmbedding interface."""
        return self.embed_text(text)
