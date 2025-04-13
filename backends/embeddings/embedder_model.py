import uuid
from typing import List, Optional
from chromadb import Collection
from sentence_transformers import SentenceTransformer


# How to use local embeddings: https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/
# All models: https://huggingface.co/models?library=sentence-transformers
# Rankings: https://huggingface.co/spaces/mteb/leaderboard
class EMBEDDER_MODEL:
    """Create vector embeddings of text."""

    def __init__(self, model: str = None, cache_path: str = None):
        embedding_model = (
            # "sentence-transformers/all-MiniLM-L6-v2"
            # "BAAI/bge-small-en-v1.5"
            # "BAAI/bge-large-en"
            # "thenlper/gte-large"
            # "thenlper/gte-small"
            "thenlper/gte-base"  # what we have historically been using
            # "intfloat/multilingual-e5-large-instruct" # @TODO currently best performer
        )
        self.model = model or embedding_model
        self.cache = cache_path

    def embed(self, text: str) -> List[float]:
        embedding_model = SentenceTransformer(self.model, cache_folder=self.cache)
        return embedding_model.encode(text, normalize_embeddings=True).tolist()

    # @TODO Use this for knowledge base actions when uploading documents
    def add_documents(
        self,
        docs: List[str],
        collection: Collection,
        metadata: Optional[List[dict]] = None,
    ):
        ids = [str(uuid.uuid4()) for _ in docs]
        embeddings = [self.embed_fn(doc) for doc in docs]
        metadatas = metadata if metadata else [None for _ in docs]
        collection.add(
            documents=docs, embeddings=embeddings, metadatas=metadatas, ids=ids
        )

    # @TODO Can we use storage.py for this without needing llama-index?
