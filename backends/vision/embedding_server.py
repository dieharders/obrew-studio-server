"""
Manages llama-server process for image embeddings.
Uses llama-server with --embedding flag to create embeddings from images.
"""

import os
import asyncio
import platform
import subprocess
from typing import List, Optional
import httpx
from core import common

LOG_PREFIX = f"{common.bcolors.OKCYAN}[IMAGE-EMBED-SERVER]{common.bcolors.ENDC}"


# This is a stop-gap to support vision based embeddings for RAG.
# Since there is no dedicated binary for this, we must use the llmaa-server binary.
class EmbeddingServer:
    """Manages llama-server process for multimodal image embeddings."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: str,
        port: int = 8081,
        n_gpu_layers: int = 99,
        n_threads: int = 4,
        n_ctx: int = 2048,
    ):
        self.process: Optional[asyncio.subprocess.Process] = None
        self.port = port
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self._is_ready = False

        # Get binary path
        deps_path = common.dep_path()
        binary_name = (
            "llama-server.exe" if platform.system() == "Windows" else "llama-server"
        )
        self.binary_path = os.path.join(deps_path, "servers", "llama.cpp", binary_name)

    @property
    def is_ready(self) -> bool:
        """Check if server is ready to accept requests."""
        return self._is_ready and self.process is not None

    @property
    def base_url(self) -> str:
        """Get the base URL for the embedding server."""
        return f"http://localhost:{self.port}"

    async def start(self) -> bool:
        """
        Start llama-server with --embedding flag.
        Returns True if server started successfully.
        """
        # Validate files exist
        if not os.path.exists(self.binary_path):
            raise FileNotFoundError(
                f"llama-server binary not found at {self.binary_path}"
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        if not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(f"mmproj file not found at {self.mmproj_path}")

        # Build command
        cmd = [
            self.binary_path,
            "-m",
            self.model_path,
            "--mmproj",
            self.mmproj_path,
            "--embedding",
            "--port",
            str(self.port),
            "-ngl",
            str(self.n_gpu_layers),
            "-t",
            str(self.n_threads),
            "-c",
            str(self.n_ctx),
        ]

        print(f"{LOG_PREFIX} Starting embedding server on port {self.port}", flush=True)
        print(f"{LOG_PREFIX} Command: {' '.join(cmd)}", flush=True)

        # Start process
        creation_kwargs = {}
        if platform.system() == "Windows":
            creation_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **creation_kwargs,
            )

            # Wait for server to be ready
            self._is_ready = await self._wait_for_ready(timeout=60)

            if self._is_ready:
                print(f"{LOG_PREFIX} Embedding server is ready", flush=True)
                return True
            else:
                print(f"{LOG_PREFIX} Server failed to become ready", flush=True)
                await self.stop()
                return False

        except Exception as e:
            print(f"{LOG_PREFIX} Failed to start server: {e}", flush=True)
            await self.stop()
            raise

    async def _wait_for_ready(self, timeout: int = 60) -> bool:
        """
        Wait for server to be ready by polling health endpoint.
        Returns True if server is ready within timeout.
        """
        health_url = f"{self.base_url}/health"
        start_time = asyncio.get_event_loop().time()

        async with httpx.AsyncClient() as client:
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    response = await client.get(health_url, timeout=2.0)
                    if response.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass

                # Check if process died
                if self.process and self.process.returncode is not None:
                    stderr = await self.process.stderr.read()
                    print(
                        f"{LOG_PREFIX} Server process died: {stderr.decode()}",
                        flush=True,
                    )
                    return False

                await asyncio.sleep(1)

        return False

    async def stop(self):
        """Gracefully stop the embedding server."""
        self._is_ready = False

        if self.process:
            print(f"{LOG_PREFIX} Stopping embedding server", flush=True)
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                print(f"{LOG_PREFIX} Force killing server", flush=True)
                self.process.kill()
            except Exception as e:
                print(f"{LOG_PREFIX} Error stopping server: {e}", flush=True)
            finally:
                self.process = None

    async def embed_image(self, image_base64: str, prompt: str = "") -> List[float]:
        """
        Send image to embedding endpoint and return embedding vector.

        Args:
            image_base64: Base64 encoded image data
            prompt: Optional text prompt to include with image

        Returns:
            List of floats representing the embedding vector
        """
        if not self.is_ready:
            raise RuntimeError("Embedding server is not ready")

        url = f"{self.base_url}/embeddings"

        # Build request payload
        # Format based on llama.cpp multimodal embedding API
        # The content must contain image placeholder [img-N] and image_data array with matching id
        content = f"[img-1]{f' {prompt}' if prompt else ''}"
        payload = {
            "content": content,
            "image_data": [
                {"id": 1, "data": image_base64}
            ]
        }

        # @TODO Dont call http, send commands directly to the binary process for the server
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    json=payload,
                    timeout=60.0,  # Image encoding can take time
                )
                response.raise_for_status()

                data = response.json()

                # Handle different response formats
                if "embedding" in data:
                    return data["embedding"]
                elif isinstance(data, list) and len(data) > 0:
                    # Some versions return array of embeddings
                    if isinstance(data[0], dict) and "embedding" in data[0]:
                        return data[0]["embedding"]
                    elif isinstance(data[0], list):
                        return data[0]

                raise ValueError(f"Unexpected response format: {data}")

            except httpx.HTTPStatusError as e:
                print(
                    f"{LOG_PREFIX} HTTP error: {e.response.status_code} - {e.response.text}",
                    flush=True,
                )
                raise
            except Exception as e:
                print(f"{LOG_PREFIX} Error embedding image: {e}", flush=True)
                raise

    async def embed_images_batch(
        self,
        images_base64: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[List[float]]:
        """
        Embed multiple images.

        Args:
            images_base64: List of base64 encoded images
            prompts: Optional list of prompts (one per image)

        Returns:
            List of embedding vectors
        """
        if prompts is None:
            prompts = [""] * len(images_base64)

        embeddings = []
        for img, prompt in zip(images_base64, prompts):
            embedding = await self.embed_image(img, prompt)
            embeddings.append(embedding)

        return embeddings

    async def get_embedding_dim(self) -> int:
        """
        Get the embedding dimension from the model.
        Sends a minimal request to determine dimension.
        """
        # This would require a test embedding - for now return 0
        # The actual dimension depends on the model
        return 0
