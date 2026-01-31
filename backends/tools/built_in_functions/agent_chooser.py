from typing import List
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Select the best option from a list. Use this tool to make a decision between multiple choices based on the given criteria."""

    options: List[str] = Field(
        ...,
        description="The list of options to choose from.",
    )
    prompt: str = Field(
        ...,
        description="The criteria or question to base the selection on.",
    )
    return_index: bool = Field(
        default=True,
        description="Whether to return the index (True) or the option text (False).",
    )
    context: str = Field(
        default="",
        description="Additional context to help make the decision.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "options": ["CSV format", "JSON format", "XML format"],
                    "prompt": "Which format is best for data interchange with web APIs?",
                    "return_index": True,
                    "context": "The data will be consumed by a REST API.",
                }
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    options = kwargs.get("options", [])
    prompt = kwargs.get("prompt")
    return_index = kwargs.get("return_index", True)
    context = kwargs.get("context", "")

    if not options:
        raise ValueError("At least one option must be provided.")

    if not prompt:
        raise ValueError("A prompt describing the selection criteria is required.")

    if len(options) < 2:
        raise ValueError("At least two options are required for selection.")

    # Build the selection prompt
    options_text = "\n".join([f"{i + 1}. {opt}" for i, opt in enumerate(options)])

    selection_prompt = f"Given the following options:\n{options_text}\n\nCriteria: {prompt}"
    if context:
        selection_prompt = f"Context:\n{context}\n\n{selection_prompt}"

    return {
        "selection_prompt": selection_prompt,
        "options": options,
        "options_count": len(options),
        "return_index": return_index,
        "status": "ready",
    }
