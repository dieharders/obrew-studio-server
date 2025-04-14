from typing import List, Callable
from chromadb.api import ClientAPI


class SimpleRAG:
    def __init__(
        self,
        client: ClientAPI,
        collection_names: List[str],
        embed_fn: Callable[[str], List[float]],
        llm_fn: Callable[[str], str],
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.db_client = client
        # @TODO loop thru and search each collection
        self.collection = self.db_client.get_or_create_collection(
            name=collection_names[0]
        )

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
