from typing import List, Tuple
from pydantic import BaseModel, Field
from core import common
from core.classes import FastAPIApp
from retrieval.simple_rag import SimpleRAG
from embeddings.main import EMBEDDING_MODEL_CACHE_PATH
from embeddings.embedder_model import EMBEDDER_MODEL
from embeddings.response_synthesis import Response_Mode
from inference.helpers import read_event_data


# Client uses `options_source` to fetch the available options to select from.
# If argument has `input_type` then it is user input, otherwise it is not shown in the FE menu.
# `options` can be used to manually inline a set of options for selection on frontend.
class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Ask a specialized knowledge agent to perform information lookup and retrieval from a data source."""
    prompt: str = Field(
        ...,
        description="The user query that is asking the agent for contextual information.",
    )
    # @TODO Remove options_source, keep field
    prompt_template: str = Field(
        ...,
        description="Name of the template to use when prompting the agent.",
        input_type="options-sel",
        placeholder="Select a template",
        options_source="retrieval-template",
    )
    similarity_top_k: int = Field(
        ...,
        description="Number of top results from knowledge base to use when ranking similarity.",
        input_type="text",
        min_value=1,
        default_value=3,
    )
    strategy: str = Field(
        ...,
        description="Name of the method to use when synthesizing an answer from contextual results.",
        input_type="options-sel",
        placeholder="Select strategy",
        default_value="refine",
        options=list(Response_Mode.__members__.values()),
    )
    # @TODO Remove options_source, keep field
    memories: List[str] = Field(
        ...,
        description="A list of collections to retrieve context information from.",
        input_type="options-multi",
        options_source="memories",
    )
    # @TODO Remove options_source, keep field
    model: Tuple[str, str] = Field(
        ...,
        description="The LLM model to use for retrieval.",
        input_type="options-sel",
        placeholder="Select model",
        options_source="installed-models",
    )

    # Used to tell LLM how to structure its response for using this tool.
    # Only include params that the LLM is expected to fill out, the rest
    # are sent in the request.
    model_config = {
        "json_schema_extra": {
            "examples": [
                # first example for display in UI/documentation
                {
                    "prompt": "Find me contact info for the head of HR.",
                    "prompt_template": "{{user_prompt}}",
                    "strategy": "summarize",
                    "similarity_top_k": 3,
                    "memories": ["company_data"],
                    "model": ["llama2", "llama2_7b_Q4.gguf"],
                },
            ]
        }
    }


# @TODO All tools should have their agent's data passed: "model", memories, etc. Just pass one big "agent" dict.
# Or should we use the same llm as the agent?
async def main(**kwargs: Params) -> str:
    # @TODO Apply template
    # prompt = apply_template(kwargs.prompt_template, kwargs.prompt)

    # Setup embedding llm
    # @TODO Pass in the model based on the metadata from the target collection. For now we always use same model.
    embedder = EMBEDDER_MODEL(cache_path=EMBEDDING_MODEL_CACHE_PATH)

    # Setup "query engine"
    similarity_top_k = kwargs["similarity_top_k"]
    app: FastAPIApp = kwargs["app"]

    # @TODO Load a seperate context for RAG
    llm = app.state.llm

    # Create a curried func, always non-streamed
    async def llm_func(prompt: str):
        print(f"{common.PRNT_RAG} Context: {prompt}", flush=True)
        # @TODO Add a system message to induce RAG behavior (dont use your internal knowledge, etc.)
        response = await llm.text_completion(request=kwargs["request"], prompt=prompt)
        # Return complete response
        content = [item async for item in response]
        data = read_event_data(content)
        return data

    # @TODO Use the RAG methodology based on "strategy" (SimpleRAG, RankerRAG, etc.) borrow code from llama-index implementation
    retriever = SimpleRAG(
        app=app,
        # @TODO Agent or Orchestrator could determine which memory to search in before calling tool
        collection_names=kwargs["memories"],
        embed_fn=embedder.embed,
        llm_fn=llm_func,
    )

    # Query
    # @TODO Replace "What is in the documents?" with kwargs["prompt"]
    result = await retriever.query(
        question="What is in the documents?", top_k=similarity_top_k
    )
    answer = result.get("text")
    return answer
