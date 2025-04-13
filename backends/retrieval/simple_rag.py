from typing import List, Callable
from embeddings.storage import get_vector_db_client

# import chromadb
# from chromadb.config import Settings


class SimpleRAG:
    def __init__(
        self,
        app,
        collection_names: List[str],
        embed_fn: Callable[[str], List[float]],
        llm_fn: Callable[[str], str],
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.client = get_vector_db_client(app)
        # @TODO loop thru and search each collection
        self.collection = self.client.get_or_create_collection(name=collection_names[0])

    def query(self, question: str, top_k: int = 5) -> str:
        query_embedding = self.embed_fn(question)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        documents = results.get("documents", [[]])[0]
        context_str = "\n".join(documents)
        # @TODO Use the prompt_template here
        prompt = f"Answer the question using the context below:\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
        return self.llm_fn(prompt)
