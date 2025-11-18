import os
import subprocess
import json
import tempfile
from typing import List, Any
from core import common
from core.classes import FastAPIApp
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

LOG_PREFIX = "[GGUF-EMBEDDER]"


class GGUFEmbedder(BaseEmbedding):
    """Handle GGUF embedding models using llama-embedding binary."""

    # Use PrivateAttr for attributes not part of the pydantic model
    _app: Any = PrivateAttr()
    _model_path: str = PrivateAttr()
    _binary_path: str = PrivateAttr()
    _n_ctx: int = PrivateAttr()
    _n_batch: int = PrivateAttr()

    def __init__(
        self,
        app: FastAPIApp,
        model_path: str = None,
        embed_model: str = None,
        n_ctx: int = 2048,
        n_batch: int = None,
        **kwargs: Any,
    ):
        # Initialize BaseEmbedding
        super().__init__(model_name=embed_model or "GGUF Embedding Model", **kwargs)

        self._app = app
        self._model_path = model_path
        self._n_ctx = n_ctx if n_ctx else 0
        # Ensure n_batch >= n_ctx to avoid assertion failure
        # When n_ctx is 0 or None, also set n_batch to 0 (use model defaults)
        if n_ctx and n_ctx > 0:
            self._n_batch = n_batch if n_batch and n_batch >= n_ctx else n_ctx
        else:
            self._n_batch = 0

        # Get path to llama-embedding binary
        deps_path = common.dep_path()
        binary_name = "llama-embedding.exe" if os.name == "nt" else "llama-embedding"
        self._binary_path = os.path.join(deps_path, "servers", "llama.cpp", binary_name)

        # Verify binary exists
        if not os.path.exists(self._binary_path):
            raise FileNotFoundError(
                f"llama-embedding binary not found at {self._binary_path}. "
                "Please restart the server to download required binaries."
            )

        print(f"{LOG_PREFIX} Initialized with model: {self.model_name}", flush=True)
        print(f"{LOG_PREFIX} Binary path: {self._binary_path}", flush=True)
        print(f"{LOG_PREFIX} Model path: {self._model_path}", flush=True)
        if self._n_ctx > 0:
            print(
                f"{LOG_PREFIX} Context size: {self._n_ctx}, Batch size: {self._n_batch}",
                flush=True,
            )
        else:
            print(
                f"{LOG_PREFIX} Using model's default context and batch size", flush=True
            )

    @classmethod
    def class_name(cls) -> str:
        return "GGUFEmbedder"

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Internal method to get embeddings (required by BaseEmbedding).

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector
        """
        return self.embed_text(text, normalize=True)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version (required by BaseEmbedding)."""
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding (required by BaseEmbedding)."""
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version (required by BaseEmbedding)."""
        return self._get_query_embedding(query)

    def embed_text(self, text: str, normalize: bool = True) -> List[float]:
        """
        Create vector embeddings from text using llama-embedding binary.

        Args:
            text: Input text to embed
            normalize: Whether to normalize the embedding vector

        Returns:
            List of floats representing the embedding vector
        """
        if not self._model_path or not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Model file not found at {self._model_path}. "
                "Please ensure the GGUF embedding model is downloaded."
            )

        # Create a temporary file for the text input
        # This avoids command-line parsing issues with special characters, newlines, etc.
        temp_file = None
        try:
            print(
                f"{LOG_PREFIX} Embedding text of length {len(text)} characters",
                flush=True,
            )

            # Write text to temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            )
            temp_file.write(text)
            temp_file.close()

            # Build command
            # Format: ./llama-embedding -m model.gguf -f temp_file.txt
            # Note: batch-size must be >= ctx-size to avoid assertion failure
            cmd = [
                self._binary_path,
                "-m",
                self._model_path,
                "-f",
                temp_file.name,  # Use file instead of -p for better handling of special chars
            ]

            # Only add context and batch size if they are specified (> 0)
            # When 0, let llama-embedding use the model's default context length
            if self._n_ctx and self._n_ctx > 0:
                cmd.extend(["-c", str(self._n_ctx)])
                # Also set batch size to match context to avoid assertion failure
                if self._n_batch and self._n_batch > 0:
                    cmd.extend(["-b", str(self._n_batch)])
                else:
                    cmd.extend(["-b", str(self._n_ctx)])

            # Optionally disable verbose logging
            cmd.append("--log-disable")

            # Add normalization parameter with proper value
            # -1=none, 0=max absolute, 2=L2 norm (default)
            if normalize:
                cmd.extend(["--embd-normalize", "2"])
            else:
                cmd.extend(["--embd-normalize", "-1"])

            print(
                f"{LOG_PREFIX} Running embedding command: {' '.join(cmd)}", flush=True
            )

            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,  # 60 second timeout
            )

            print(f"{LOG_PREFIX} Command completed successfully", flush=True)

            # Parse output
            # llama-embedding outputs embeddings as JSON array on stdout
            output = result.stdout.strip()

            # The output format is typically: embedding <array>
            # We need to extract just the array part
            if "embedding" in output:
                # Find the JSON array in the output
                start_idx = output.find("[")
                end_idx = output.rfind("]") + 1

                if start_idx >= 0 and end_idx > start_idx:
                    json_str = output[start_idx:end_idx]
                    embeddings = json.loads(json_str)

                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        print(
                            f"{LOG_PREFIX} Generated embedding with {len(embeddings)} dimensions",
                            flush=True,
                        )
                        return embeddings

            # Fallback: try to parse entire output as JSON
            try:
                embeddings = json.loads(output)
                if isinstance(embeddings, list):
                    print(
                        f"{LOG_PREFIX} Generated embedding with {len(embeddings)} dimensions",
                        flush=True,
                    )
                    return embeddings
            except json.JSONDecodeError:
                pass

            raise ValueError(f"Failed to parse embedding output: {output[:100]}...")

        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Embedding generation timed out after 60 seconds for text of length {len(text)}"
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"llama-embedding failed with exit code {e.returncode}."
            stderr_output = e.stderr if e.stderr else "(empty)"
            stdout_output = e.stdout if e.stdout else "(empty)"
            error_msg += f"\nstderr: {stderr_output[:2000]}"
            error_msg += f"\nstdout: {stdout_output[:2000]}"
            error_msg += f"\nCommand: {' '.join(cmd)}"
            print(f"{LOG_PREFIX} {error_msg}", flush=True)
            raise RuntimeError(error_msg)
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as cleanup_err:
                    print(
                        f"{LOG_PREFIX} Warning: Failed to delete temp file: {cleanup_err}",
                        flush=True,
                    )

    def get_text_embedding(self, text: str) -> List[float]:
        """Alias for embed_text for compatibility with HuggingFaceEmbedding interface."""
        return self.embed_text(text, normalize=True)
