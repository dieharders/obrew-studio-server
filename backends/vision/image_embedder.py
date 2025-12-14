"""
High-level interface for image embedding.
Manages the embedding server and provides methods to embed images.
"""

import os
import base64
from pathlib import Path
from typing import List, Optional
from core import common
from core.classes import FastAPIApp
from .embedding_server import EmbeddingServer

LOG_PREFIX = f"{common.bcolors.OKCYAN}[IMAGE-EMBEDDER]{common.bcolors.ENDC}"


class ImageEmbedder:
    """High-level interface for creating image embeddings."""

    def __init__(self, app: FastAPIApp):
        self.app = app
        self.server: Optional[EmbeddingServer] = None
        self.model_path: Optional[str] = None
        self.mmproj_path: Optional[str] = None
        self.model_name: Optional[str] = None
        self.model_id: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        """Check if an embedding model is loaded and ready."""
        return self.server is not None and self.server.is_ready

    async def load_model(
        self,
        model_path: str,
        mmproj_path: str,
        model_name: str = None,  # @TODO Do we rly need this?
        model_id: str = None,  # @TODO Do we rly need this?
        port: int = 8081,
        n_gpu_layers: int = 999,
        n_threads: int = 4,  # @TODO set to auto
        n_ctx: int = 2048,  # @TODO Do we rly need this, can set to max?
    ) -> bool:
        """
        Load a multimodal embedding model.

        Args:
            model_path: Path to the GGUF model file
            mmproj_path: Path to the mmproj (multimodal projector) file
            model_name: Friendly name for the model
            model_id: Model identifier
            port: Port to run the embedding server on
            n_gpu_layers: Number of layers to offload to GPU
            n_threads: Number of CPU threads to use
            n_ctx: Context window size

        Returns:
            True if model loaded successfully
        """
        # Unload existing model first
        if self.server:
            await self.unload()

        # Validate paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(mmproj_path):
            raise FileNotFoundError(f"mmproj file not found: {mmproj_path}")

        print(f"{LOG_PREFIX} Loading embedding model: {model_path}", flush=True)
        print(f"{LOG_PREFIX} mmproj: {mmproj_path}", flush=True)

        # Store model info
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.model_name = model_name or Path(model_path).stem
        self.model_id = model_id or self.model_name

        # Create and start embedding server
        self.server = EmbeddingServer(
            model_path=model_path,
            mmproj_path=mmproj_path,
            port=port,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            n_ctx=n_ctx,
        )

        success = await self.server.start()

        if success:
            print(
                f"{LOG_PREFIX} Model loaded successfully: {self.model_name}", flush=True
            )
        else:
            print(f"{LOG_PREFIX} Failed to load model", flush=True)
            self.server = None

        return success

    async def unload(self):
        """Unload the current embedding model and stop the server."""
        if self.server:
            print(f"{LOG_PREFIX} Unloading model: {self.model_name}", flush=True)
            await self.server.stop()
            self.server = None

        self.model_path = None
        self.mmproj_path = None
        self.model_name = None
        self.model_id = None

    async def embed_image_from_path(
        self,
        image_path: str,
        prompt: str = "",
    ) -> List[float]:
        """
        Create embedding for an image from file path.

        Args:
            image_path: Path to the image file
            prompt: Optional text prompt to include

        Returns:
            List of floats representing the embedding vector
        """
        # @TODO Perhaps we should auto load the model here instead and pass the model to load to this.
        if not self.is_loaded:
            raise RuntimeError("No embedding model loaded. Call load_model() first.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        return await self.server.embed_image(image_base64, prompt)

    async def embed_image_from_base64(
        self,
        image_base64: str,
        prompt: str = "",
    ) -> List[float]:
        """
        Create embedding for a base64 encoded image.

        Args:
            image_base64: Base64 encoded image data
            prompt: Optional text prompt to include

        Returns:
            List of floats representing the embedding vector
        """
        if not self.is_loaded:
            raise RuntimeError("No embedding model loaded. Call load_model() first.")

        # Strip data URL prefix if present
        if image_base64.startswith("data:"):
            # Extract base64 portion after comma
            image_base64 = image_base64.split(",", 1)[1]

        return await self.server.embed_image(image_base64, prompt)

    async def embed_images_batch(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
    ) -> List[List[float]]:
        """
        Create embeddings for multiple images.

        Args:
            image_paths: List of image file paths
            prompts: Optional list of prompts (one per image)

        Returns:
            List of embedding vectors
        """
        if not self.is_loaded:
            raise RuntimeError("No embedding model loaded. Call load_model() first.")

        # Encode all images
        images_base64 = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            with open(path, "rb") as f:
                image_data = f.read()
            images_base64.append(base64.b64encode(image_data).decode("utf-8"))

        return await self.server.embed_images_batch(images_base64, prompts)

    def get_model_info(self) -> dict:
        """Get information about the currently loaded model."""
        if not self.is_loaded:
            return {
                "loaded": False,
                "model_name": None,
                "model_id": None,
                "model_path": None,
                "mmproj_path": None,
            }

        return {
            "loaded": True,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "mmproj_path": self.mmproj_path,
            "server_port": self.server.port if self.server else None,
        }
