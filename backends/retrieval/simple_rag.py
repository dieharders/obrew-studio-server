from typing import List, Callable
from chromadb import Collection


class SimpleRAG:
    def __init__(
        self,
        collections: List[Collection],
        embed_fn: Callable[[str], List[float]],
        llm_fn: Callable[[str], str],
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.collections = collections

    def query(self, question: str, top_k: int = 5) -> str:
        query_embedding = self.embed_fn(question)
        # @TODO loop thru and search each collection (query_embeddings)
        collection = self.collections[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        context_str = "\n".join(documents)
        # @TODO Use the prompt_template here
        prompt = f"Answer the question using the context below:\n\n{context_str}\n\nQuestion: {question}\nAnswer:"
        return self.llm_fn(prompt)
