import os
import json
import asyncio
from typing import Awaitable, List, Callable, Union
from chromadb import Collection
from inference.classes import AgentOutput
from inference.helpers import apply_query_template
from core import common


class RAG:
    def __init__(
        self,
        collection: Collection,
        embed_fn: Union[
            Callable[[str], List[float]], Callable[[str], Awaitable[List[float]]]
        ],
        llm_fn: Awaitable[AgentOutput],
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.collection = collection
        self.system_message_override = "You are a helpful assistant. Use the provided context to answer the user's question.\n\nIf you don't know the answer based on the context, say you don't know. Do not make up an answer."
        self.retrieval_templates = dict()

        # Get the file for the retrieval templates
        try:
            path = common.dep_path(os.path.join("public", "retrieval_templates.json"))
            with open(path, "r") as file:
                self.retrieval_templates = json.load(file)
        except Exception as err:
            raise Exception(f"Error finding prompt format templates: {err}")

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding, handling both sync and async embed_fn."""
        result = self.embed_fn(text)
        # If embed_fn returned a coroutine, await it
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def query(self) -> AgentOutput:
        return {"text": ""}


class SimpleRAG(RAG):
    def __init__(self, collection, embed_fn, llm_fn):
        super().__init__(collection, embed_fn, llm_fn)

    async def query(
        self,
        question: str,
        system_message: str = None,
        template: str = None,
        top_k: int = 5,
        expand_context: bool = True,  # Use original_text from metadata (sentence window context) instead of just the matched chunk
    ) -> AgentOutput:
        # Use _get_embedding to handle both sync and async embed functions
        query_embedding = await self._get_embedding(question)
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        # Expand context for sentence window chunks if available
        if expand_context and metadatas:
            expanded_documents = []
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                # Use original_text (with window context) if available
                expanded_text = metadata.get("original_text", doc)
                expanded_documents.append(expanded_text)
            documents = expanded_documents

        template_override = self.retrieval_templates.get("CONTEXT_ONLY").get("text")
        prompt_template = template or template_override
        prompt = apply_query_template(
            template=prompt_template,
            query=question,
            context_list=documents,
        )
        sys_msg = system_message or self.system_message_override
        return await self.llm_fn(prompt=prompt, system_message=sys_msg)
