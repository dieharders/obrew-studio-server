from typing import List, Tuple
from pydantic import BaseModel, Field
from embeddings.response_synthesis import Response_Mode


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
    memories: List[str] = Field(
        ...,
        description="A list of collections to retrieve context information from.",
        input_type="options-multi",
        options_source="memories",
    )
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


async def main(**kwargs: Params) -> str:
    # prompt = apply_template(args.prompt_template, args.prompt)
    # @TODO RAG logic here...
    # retriever = RAG(options)
    # result = retriever.query(prompt)
    result = ""
    return result
