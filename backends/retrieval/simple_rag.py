from typing import List, Callable
from chromadb import Collection
from inference.helpers import apply_query_template


class SimpleRAG:
    def __init__(
        self,
        collection: Collection,
        embed_fn: Callable[[str], List[float]],
        llm_fn: Callable[[str], str],
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.collection = collection

    def query(
        self, question: str, system_message: str, template: str, top_k: int = 5
    ) -> str:
        query_embedding = self.embed_fn(question)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        documents = results.get("documents", [[]])[0]
        # @TODO Override the template based "strategy"? Otherwise user has to know which templates go with which strategy.
        prompt = apply_query_template(
            template=template, query=question, context_list=documents
        )
        return self.llm_fn(prompt, system_message)
