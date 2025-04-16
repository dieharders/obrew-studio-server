from typing import List, Callable
from chromadb import Collection
from inference.helpers import apply_query_template


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

    def query(
        self, question: str, system_message: str, template: str, top_k: int = 5
    ) -> str:
        query_embedding = self.embed_fn(question)
        # @TODO loop thru and search each collection (query_embeddings)
        collection = self.collections[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        prompt = apply_query_template(
            template=template, query=question, context_list=documents
        )
        return self.llm_fn(prompt, system_message)
