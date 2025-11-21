import os
import subprocess
import json
import tempfile
from typing import List, Any
from core import common
from core.classes import FastAPIApp
from llama_index.core.embeddings import BaseEmbedding
from pydantic.v1 import PrivateAttr

LOG_PREFIX = "[GGUF-EMBEDDER]"


class GGUFEmbedder(BaseEmbedding):
    """Handle GGUF embedding models using llama-cli binary."""

    # Use PrivateAttr for attributes not part of the pydantic model
    _app: Any = PrivateAttr(default=None)
    _model_path: str = PrivateAttr(default=None)
    _binary_path: str = PrivateAttr(default=None)

    def __init__(
        self,
        app: FastAPIApp,
        model_path: str = None,
        embed_model: str = None,
        **kwargs: Any,
    ):
        # Initialize BaseEmbedding
        super().__init__(model_name=embed_model or "GGUF Embedding Model", **kwargs)

        self._app = app
        self._model_path = model_path

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
            # Format: ./llama-embedding -m /embed_models/models--nomic-ai--nomic-embed-text-v1.5-GGUF/snapshots/0188c9bf409793f810680a5a431e7b899c46104c/nomic-embed-text-v1.5.Q8_0.gguf -p 'Hello World!' --pooling mean --n-gpu-layers 99 --embd-output-format array --embd-normalize 2
            # Note: batch-size must be >= ctx-size to avoid assertion failure
            cmd = [
                self._binary_path,
                "-m",
                self._model_path,
                # "-p",
                # "Hello World!",
                "-f",
                temp_file.name,  # Use file instead of -p for better handling of special chars
            ]
            cmd.extend(["--pooling", "mean"])
            cmd.extend(["--n-gpu-layers", "99"])

            # Set output format to JSON array for easy parsing
            cmd.extend(["--embd-output-format", "array"])

            # Add normalization parameter with proper value
            # -1=none, 0=max absolute, 2=L2 norm (default)
            if normalize:
                cmd.extend(["--embd-normalize", "2"])
            else:
                cmd.extend(["--embd-normalize", "-1"])

            # NOTE: Do NOT use --log-disable as it suppresses the embedding output!
            # Performance stats go to stderr, so they don't interfere with stdout parsing

            print(
                f"{LOG_PREFIX} Running GGUF embedding command: {' '.join(cmd)}",
                flush=True,
            )

            # Run the command
            # Hide console window on all platforms (especially important in production)
            # CREATE_NO_WINDOW is Windows-specific, gracefully falls back to 0 on other platforms
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,  # 60 second timeout
                stdin=subprocess.DEVNULL,  # Prevent any input prompts (no window)
                creationflags=creation_flags,
            )

            print(
                f"{LOG_PREFIX} Command completed successfully.",
                flush=True,
            )

            # Parse output
            # With --embd-output-format array, we get: [[embedding_vector]]
            # Performance stats go to stderr, so stdout contains only the JSON array
            output = result.stdout.strip()

            try:
                # Parse JSON array directly from stdout
                result_array = json.loads(output)

                # The format is [[embedding]], so we need the first (and only) element
                if isinstance(result_array, list) and len(result_array) > 0:
                    embeddings = result_array[0]  # Get first embedding from array

                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        print(
                            f"{LOG_PREFIX} Generated embedding with {len(embeddings)} dimensions",
                            flush=True,
                        )
                        return embeddings
                    else:
                        raise ValueError(
                            f"Invalid embedding format: expected list of floats, got {type(embeddings)}"
                        )
                else:
                    raise ValueError(
                        f"Invalid output format: expected non-empty array, got {type(result_array)}"
                    )

            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse JSON output: {e}. Extracted JSON: {output[:200]}"
                )

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
