from typing import List
from chromadb import Collection
from pydantic import BaseModel, Field
from core import common
from core.classes import FastAPIApp
from retrieval.rag import SimpleRAG
from embeddings.vector_storage import Vector_Storage
from embeddings.embedder import Embedder
from embeddings.response_synthesis import Response_Mode
from inference.helpers import read_event_data


# Client uses `options_source` to fetch the available options to select from.
# If argument has `input_type` then it is user input, otherwise it is not shown in the FE menu.
# `options` can be used to manually inline a set of options for selection on frontend.
class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Ask a specialized knowledge agent to perform information lookup and retrieval from a data source."""

    # prompt: str = Field(
    #     ...,
    #     description="The user prompt that asks the agent for contextual information.",
    #     llm_not_required=True,
    # )
    # system_message: str = Field(
    #     ...,
    #     description="The agent's system message instructs it how to handle the contextual information provided.",
    #     # @TODO Maybe we can allow user to specify this?
    #     llm_not_required=True,
    # )
    # prompt_template: str = Field(
    #     ...,
    #     description="The agent's prompt template is used to structure its' response and influence what information is retrieved from the provided context.",
    #     input_type="options-sel",
    #     placeholder="Select a template",
    #     options_source="retrieval-template",
    #     llm_not_required=True,
    # )
    similarity_top_k: int = Field(
        ...,
        description="Specify the number of top results from the knowledge base to use when ranking similarity.",
        input_type="text",
        min_value=1,
        default_value=3,
        llm_not_required=True,
    )
    strategy: str = Field(
        ...,
        description="Choose the method to use when synthesizing an answer from the retrieved context.",
        input_type="options-sel",
        placeholder="Select strategy",
        default_value="refine",
        options=list(Response_Mode.__members__.values()),
        llm_not_required=True,
    )
    # memories: List[str] = Field(
    #     ...,
    #     description="Access the agent's memories to retrieve context information.",
    #     # input_type="options-multi",
    #     # options_source="memories",
    #     llm_not_required=True,
    # )
    # model: Tuple[str, str] = Field(
    #     ...,
    #     description="Use the agent's LLM model [model id, model filename] for retrieval.",
    #     # input_type="options-sel",
    #     # placeholder="Select model",
    #     # options_source="installed-models",
    #     llm_not_required=True,
    # )

    # Used to tell LLM how to structure its response for using this tool.
    # Only include params that the LLM is expected to fill out, the rest
    # are sent in the request.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    # "prompt": "Find me contact info for the head of HR.",
                    # "system_message": "Only use knowledge taken from the provided context.",
                    # "prompt_template": "{{user_prompt}}",
                    "strategy": "summarize",
                    "similarity_top_k": 3,
                    # "memories": ["private_data"],
                    # "model": ["llama2", "llama2_7b_Q4.gguf"],
                },
            ]
        }
    }


# prompt_template and system_message are overriden by RAG implementations
async def main(**kwargs: Params) -> str:
    generate_kwargs = kwargs.get("generate_kwargs")
    model_init_kwargs = kwargs.get("model_init_kwargs")
    template = kwargs.get("prompt_template")
    system_message = kwargs.get("system_message")
    app: FastAPIApp = kwargs.get("app")
    query = kwargs.get("prompt")
    similarity_top_k = kwargs.get("similarity_top_k")
    collection_names = kwargs.get("memories", [])
    if len(collection_names) == 0:
        raise Exception("No collection names specified.")
    strategy = kwargs.get("strategy")

    # Setup embedding llm
    vector_storage = Vector_Storage(app=app)

    # @TODO Load a separate context for RAG
    llm = app.state.llm

    # Create a curried func, always non-streamed
    async def llm_func(prompt: str, system_message: str):
        print(f"{common.PRNT_RAG} Context:\n{prompt}", flush=True)
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

    # @TODO Agent or Orchestrator could loop thru each collection's metadata and determine which one to select before calling tool
    # OR
    # We could make embeddings from each collection's metadata and compare those with query embedding to determine which is closest (most relevant).
    selected_collection = collections[0]

    # Pass in the embed model based on the metadata from the target collection
    embed_model_name = selected_collection.metadata.get("embedding_model")
    embedder = Embedder(app=app, embed_model=embed_model_name)

    # Use the RAG methodology (SimpleRAG, RankerRAG, etc.) based on "strategy" borrow code from llama-index implementation
    match strategy:
        case Response_Mode.CONTEXT_ONLY:
            retriever = SimpleRAG(
                collection=selected_collection,
                embed_fn=embedder.embed_text,
                llm_fn=llm_func,
            )
        case _:
            retriever = SimpleRAG(
                collection=selected_collection,
                embed_fn=embedder.embed_text,
                llm_fn=llm_func,
            )

    # Query
    result = await retriever.query(
        question=query,
        # system_message=system_message, # overridden internally
        # template=template, # overridden internally
        top_k=similarity_top_k,
    )
    answer = result.get("text")
    return answer
