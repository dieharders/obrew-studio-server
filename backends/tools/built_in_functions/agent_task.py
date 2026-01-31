from typing import Literal, Union
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Execute a single instruction and return a structured result. Use this tool to perform one focused task with optional structured output."""

    prompt: str = Field(
        ...,
        description="The instruction or task to execute.",
    )
    context: str = Field(
        default="",
        description="Additional context or information to help execute the task.",
    )
    output_format: Literal["text", "json", "list"] = Field(
        default="text",
        description="The format for the output response.",
        options=["text", "json", "list"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Summarize the key points from this document.",
                    "context": "The document discusses project management best practices...",
                    "output_format": "list",
                }
            ]
        }
    }


async def main(**kwargs: Params) -> Union[str, dict, list]:
    prompt = kwargs.get("prompt")
    context = kwargs.get("context", "")
    output_format = kwargs.get("output_format", "text")

    if not prompt:
        raise ValueError("A prompt is required for the task.")

    # Build the full prompt with context
    full_prompt = prompt
    if context:
        full_prompt = f"Context:\n{context}\n\nTask:\n{prompt}"

    # Add format instructions based on output_format
    format_instructions = {
        "text": "Provide your response as plain text.",
        "json": "Provide your response as valid JSON.",
        "list": "Provide your response as a numbered list.",
    }

    if output_format in format_instructions:
        full_prompt += f"\n\n{format_instructions[output_format]}"

    # The actual LLM call would be handled by the agent system
    # This tool returns the structured prompt for the agent to process
    return {
        "task_prompt": full_prompt,
        "output_format": output_format,
        "status": "ready",
    }
