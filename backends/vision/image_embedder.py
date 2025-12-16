"""
High-level interface for image embedding.
Manages the embedding server and provides methods to embed images.
"""

import os
import base64
from pathlib import Path
from typing import List, Optional, Tuple
from core import common
from core.classes import FastAPIApp
from .embedding_server import EmbeddingServer

LOG_PREFIX = f"{common.bcolors.OKCYAN}[IMAGE-EMBEDDER]{common.bcolors.ENDC}"

# Default vision embedding model configurations
vision_embedding_model_names = dict(
    gme_varco="mradermacher/GME-VARCO-VISION-Embedding-GGUF",
)
DEFAULT_VISION_EMBEDDING_MODEL = vision_embedding_model_names["gme_varco"]
VISION_EMBEDDING_MODELS_CACHE_DIR = common.app_path(
    common.VISION_EMBEDDING_MODELS_CACHE_DIR
)
DEFAULT_EMBEDDING_PORT = 8081


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

    def _find_model_files(
        self, repo_id: str = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find GGUF model and mmproj files in the vision embedding models cache.

        Args:
            repo_id: Optional specific repo ID to look for. If None, searches for any model.

        Returns:
            Tuple of (model_path, mmproj_path) or (None, None) if not found.
        """
        cache_dir = VISION_EMBEDDING_MODELS_CACHE_DIR

        if not os.path.exists(cache_dir):
            print(f"{LOG_PREFIX} Cache directory not found: {cache_dir}", flush=True)
            return None, None

        # If repo_id specified, look for that specific model
        if repo_id:
            repo_slug = repo_id.replace("/", "--")
            repo_path = os.path.join(cache_dir, f"models--{repo_slug}")
            if os.path.exists(repo_path):
                return self._find_gguf_and_mmproj_in_repo(repo_path)
            return None, None

        # Otherwise, search all repos in cache for any valid model
        for item in os.listdir(cache_dir):
            if item.startswith("models--"):
                repo_path = os.path.join(cache_dir, item)
                if os.path.isdir(repo_path):
                    model_path, mmproj_path = self._find_gguf_and_mmproj_in_repo(
                        repo_path
                    )
                    if model_path and mmproj_path:
                        return model_path, mmproj_path

        return None, None

    def _find_gguf_and_mmproj_in_repo(
        self, repo_path: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Find GGUF model and mmproj files within a HuggingFace cache repo directory.

        Args:
            repo_path: Path to the repo directory (e.g., models--owner--repo-name)

        Returns:
            Tuple of (model_path, mmproj_path) or (None, None) if not found.
        """
        snapshots_path = os.path.join(repo_path, "snapshots")
        if not os.path.exists(snapshots_path):
            return None, None

        model_path = None
        mmproj_path = None

        # Look through snapshot directories
        for snapshot_dir in os.listdir(snapshots_path):
            snapshot_full_path = os.path.join(snapshots_path, snapshot_dir)
            if not os.path.isdir(snapshot_full_path):
                continue

            # Find .gguf files
            for file in os.listdir(snapshot_full_path):
                if not file.endswith(".gguf"):
                    continue

                file_path = os.path.join(snapshot_full_path, file)
                file_lower = file.lower()

                # Identify mmproj files
                if "mmproj" in file_lower:
                    if mmproj_path is None:
                        mmproj_path = file_path
                        print(f"{LOG_PREFIX} Found mmproj: {file_path}", flush=True)
                else:
                    # Regular model file - prefer Q4_K_M quantization if available
                    if model_path is None or "q4_k_m" in file_lower:
                        model_path = file_path
                        print(f"{LOG_PREFIX} Found model: {file_path}", flush=True)

        return model_path, mmproj_path

    async def _ensure_model_loaded(self) -> None:
        """
        Ensure an embedding model is loaded, auto-loading from cache if needed.

        This method checks if an embedding model is already loaded. If not, it attempts
        to find and load a model from the vision embedding models cache directory.

        Raises:
            RuntimeError: If no model is loaded and auto-loading fails.
        """
        # Already loaded
        if self.is_loaded:
            return

        # Try to find model files in cache
        print(f"{LOG_PREFIX} Auto-loading vision embedding model...", flush=True)
        model_path, mmproj_path = self._find_model_files()

        if model_path and mmproj_path:
            success = await self.load_model(
                model_path=model_path,
                mmproj_path=mmproj_path,
                port=DEFAULT_EMBEDDING_PORT,
            )

            if success:
                print(f"{LOG_PREFIX} Model auto-loaded successfully", flush=True)
                return

            raise RuntimeError(
                f"Failed to auto-load vision embedding model from {model_path}"
            )

        # No model found in cache
        raise RuntimeError(
            f"No vision embedding model found in cache directory: {VISION_EMBEDDING_MODELS_CACHE_DIR}. "
            "Please download a vision embedding model first using /v1/vision/embed/download endpoint."
        )

    async def embed_images(
        self,
        image_paths: List[str],
        prompts: Optional[List[str]] = None,
        auto_unload: bool = True,
    ) -> List[List[float]]:
        """
        Create embeddings for one or more images.

        Args:
            image_paths: List of image file paths (can be single item)
            prompts: Optional list of prompts (one per image)
            auto_unload: Whether to unload the model after embedding (default True).
                         Set to False when embedding multiple batches sequentially
                         to avoid the overhead of reloading the model each time.

        Returns:
            List of embedding vectors
        """
        try:
            # Auto-load model if not loaded
            await self._ensure_model_loaded()

            # Encode all images
            images_base64 = []
            for path in image_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image not found: {path}")
                with open(path, "rb") as f:
                    image_data = f.read()
                images_base64.append(base64.b64encode(image_data).decode("utf-8"))

            return await self.server.embed_images_batch(images_base64, prompts)
        finally:
            if auto_unload:
                await self.unload()

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
