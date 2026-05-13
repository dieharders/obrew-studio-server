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

# Timeout for server readiness
SERVER_READY_TIMEOUT = 60


class GGUFEmbedderServer:
    """Handle GGUF embedding models using llama-server with --embedding flag."""

    def __init__(
        self,
        app: FastAPIApp,
        model_path: str = None,
        embed_model: str = None,
        pooling: str = "mean",
    ):
        self.app = app
        self.model_name = embed_model or "GGUF Embedding Model"
        self.model_path = model_path
        self.pooling = pooling or "mean"
        self.process: subprocess.Popen = None
        self._is_ready = False
        self._http_client: httpx.Client = None

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

        # Pick port via shared allocator (prevents collisions with LlamaServer).
        self.port = common.allocate_server_port()
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
            "--pooling", self.pooling,
            "--port", str(self.port),
            "-ngl", "99",
        ]

        print(f"{LOG_PREFIX} Starting embedding server on port {self.port}", flush=True)

        creation_kwargs = {}
        if platform.system() == "Windows":
            creation_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        # Set cwd to binary directory so DLLs next to the exe are found
        binary_dir = os.path.dirname(self.binary_path)

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                cwd=binary_dir,
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
        """Poll /health synchronously until the server is ready.

        This intentionally busy-polls with time.sleep() because it runs from
        background threads (via BackgroundTasks) that have no asyncio event loop.
        The sleep interval is short to detect readiness quickly.
        """
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

                time.sleep(0.5)

        return False

    def _stop_process(self):
        """Synchronously stop the server process and release resources."""
        try:
            if self._http_client and not self._http_client.is_closed:
                self._http_client.close()
            self._http_client = None
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
        finally:
            common.release_server_port(self.port)

    async def stop(self):
        """Stop the embedding server (async interface for shutdown path)."""
        self._is_ready = False
        self._stop_process()

    def _parse_embedding_response(self, data) -> List[float]:
        """Parse embedding from the various response formats llama-server can return."""
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict) and "embedding" in data[0]:
                return data[0]["embedding"]
            elif isinstance(data[0], list):
                return data[0]
        elif isinstance(data, dict):
            if "embedding" in data:
                return data["embedding"]
            elif "data" in data and isinstance(data["data"], list):
                if len(data["data"]) > 0 and "embedding" in data["data"][0]:
                    return data["data"][0]["embedding"]
        raise ValueError(f"Unexpected embedding response format: {data}")

    def _get_http_client(self) -> httpx.Client:
        """Return the persistent HTTP client, creating one if needed."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.Client()
        return self._http_client

    def _ensure_server(self):
        """Start the server if not running, or restart if the process has crashed."""
        if self._is_ready and self.process and self.process.poll() is None:
            return
        # Process died or was never started — reset and (re)start
        if self.process and self.process.poll() is not None:
            print(f"{LOG_PREFIX} Server process crashed, restarting...", flush=True)
        self._is_ready = False
        result = self._start_server()
        if not result:
            raise RuntimeError("Failed to start embedding server")

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

        self._ensure_server()

        print(
            f"{LOG_PREFIX} Embedding text of length {len(text)} characters",
            flush=True,
        )

        url = f"{self.base_url}/embeddings"
        payload = {"content": text}
        client = self._get_http_client()

        try:
            response = client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()

            data = response.json()
            raw_embedding = self._parse_embedding_response(data)
            print(
                f"{LOG_PREFIX} Generated embedding with {len(raw_embedding)} dimensions",
                flush=True,
            )
            return raw_embedding

        except (httpx.ConnectError, httpx.RemoteProtocolError):
            # Server may have crashed mid-request — restart once and retry
            print(f"{LOG_PREFIX} Connection lost, restarting server and retrying...", flush=True)
            self._is_ready = False
            self._ensure_server()
            # Get a fresh client in case the old connection is stale
            client = self._get_http_client()
            response = client.post(url, json=payload, timeout=60.0)
            response.raise_for_status()
            data = response.json()
            return self._parse_embedding_response(data)
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Embedding server returned error: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}")

    def get_text_embedding(self, text: str) -> List[float]:
        """Alias for embed_text for compatibility with HuggingFaceEmbedding interface."""
        return self.embed_text(text)
