from typing import List, Callable, Union, Awaitable
from chromadb import Collection
from pydantic import BaseModel, Field
from core import common
from core.classes import FastAPIApp
from retrieval.rag import SimpleRAG
from embeddings.vector_storage import Vector_Storage
from embeddings.embedder import Embedder
from embeddings.rag_response_synthesis import RESPONSE_SYNTHESIS_MODES
from inference.helpers import read_event_data
from vision.image_embedder import ImageEmbedder

# Type alias for embed functions (sync or async)
EmbedFnType = Union[
    Callable[[str], List[float]], Callable[[str], Awaitable[List[float]]]
]


def _create_vision_embed_fn(vision_embedder: ImageEmbedder) -> EmbedFnType:
    """
    Create an async embed function for the vision embedder.
    Defined outside loop to avoid closure issues with loop variables.
    """

    async def vision_embed_fn(text: str) -> List[float]:
        return await vision_embedder.embed_query_text(text, auto_unload=False)

    return vision_embed_fn


# Client uses `options_source` to fetch the available options to select from.
# If argument has `input_type` then it is user input, otherwise it is not shown in the FE menu.
# `options` can be used to manually inline a set of options for selection on frontend.
class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Ask a specialized knowledge agent to perform information lookup and retrieval from a data source."""

    prompt: str = Field(
        ...,
        description="The user prompt that asks the agent for contextual information.",
        llm_not_required=True,
    )

    # Used to tell LLM how to structure its response for using this tool.
    # Only include params that the LLM is expected to fill out, the rest
    # are sent in the request.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Find me contact info for the head of HR.",
                    "system_message": "Only use knowledge taken from the provided context.",
                    # "prompt_template": "{{user_prompt}}",
                    "strategy": "refine",
                    "similarity_top_k": 3,
                },
            ]
        }
    }


# prompt_template and system_message are overriden by RAG implementations
async def main(**kwargs: Params) -> str:
    generate_kwargs = kwargs.get("generate_kwargs")
    model_init_kwargs = kwargs.get("model_init_kwargs")
    # template = kwargs.get("prompt_template")
    system_message = kwargs.get("system_message")
    app: FastAPIApp = kwargs.get("app")
    query = kwargs.get("prompt")
    if not query:
        print(f"{common.PRNT_RAG} Warning: query does not exist.", flush=True)
    similarity_top_k = kwargs.get("similarity_top_k")
    strategy = kwargs.get("strategy")
    collection_names = kwargs.get("memories", [])
    if len(collection_names) == 0:
        raise Exception(
            "Retrieval Tool: Please provide collection names for memory retrieval."
        )

    # Setup embedding llm
    vector_storage = Vector_Storage(app=app)

    # @TODO Load a separate context for RAG. I think we may need a retrieval class if we dont already.
    llm = app.state.llm

    # Create a curried func, always non-streamed
    async def llm_func(prompt: str, system_message: str):
        print(f"{common.PRNT_RAG} Context:\n{prompt}", flush=True)
        # @TODO Why do we pass prompt and sys msg here and also to retriever.query() ?
        response = await llm.text_completion(
            request=kwargs.get("request"), prompt=prompt, system_message=system_message
        )
        # Return complete response
        content = [item async for item in response]
        data = read_event_data(content)
        return data

    # Gather list of collections from names
    collections: List[Collection] = []
    for name in collection_names:
        collection = vector_storage.db_client.get_collection(name=name)
        collections.append(collection)

    # Loop thru each collection and return results
    cumulative_answer = ""
    vision_embedder: ImageEmbedder | None = None  # Lazy-init vision embedder if needed

    try:
        for selected_collection in collections:
            # Detect collection type and use appropriate embedder
            collection_type = selected_collection.metadata.get("type", "")
            embed_model_name = selected_collection.metadata.get("embedding_model")
            embed_fn: EmbedFnType

            if collection_type == "image_embeddings":
                # Use vision embedder for image collections
                print(
                    f"{common.PRNT_RAG} Detected image collection, using vision embedder",
                    flush=True,
                )
                if vision_embedder is None:
                    vision_embedder = ImageEmbedder(app)

                # Use factory function to avoid closure issues in loop
                embed_fn = _create_vision_embed_fn(vision_embedder)
            else:
                # Use text embedder for text collections
                embedder = Embedder(app=app, embed_model=embed_model_name)
                embed_fn = embedder.embed_text

            # Use the RAG methodology (SimpleRAG, RankerRAG, etc.) based on "strategy" borrow code from llama-index implementation
            # @TODO Perform the different "strategies" (tree-summarize, etc) will be performed by app layer
            match strategy:
                case RESPONSE_SYNTHESIS_MODES.CONTEXT_ONLY.value:
                    retriever = SimpleRAG(
                        collection=selected_collection,
                        embed_fn=embed_fn,
                        llm_fn=llm_func,
                    )
                case _:
                    retriever = SimpleRAG(
                        collection=selected_collection,
                        embed_fn=embed_fn,
                        llm_fn=llm_func,
                    )

            # Query
            result = await retriever.query(
                question=query or "",
                system_message=system_message,  # overridden internally if not provided
                # @TODO We will remove this and app will pass an already built prompt (we should rly only need the user query)
                # template=template,  # overridden internally if not provided
                top_k=similarity_top_k or 5,
            )
            answer = result.get("text")
            cumulative_answer += f"\n\n{answer}"

        return cumulative_answer.strip()

    finally:
        # Cleanup: always unload vision embedder if it was used, even on exception
        if vision_embedder is not None:
            await vision_embedder.unload()
