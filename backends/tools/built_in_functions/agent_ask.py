from typing import Literal
from pydantic import BaseModel, Field


# @TODO This is a stub, pls implement fully later.
# This is useful in an agent's list of tools as it gives them the option to ask the user for clarity or decision making (planning, creating todos).
class Params(BaseModel):
    """Generate a conversational response. Use this tool for clarification, confirmation, creative answers, technical breakdowns, or presenting choices to the user."""

    prompt: str = Field(
        ...,
        description="The message or question to respond to.",
    )
    mode: Literal["clarify", "confirm", "creative", "technical", "choice"] = Field(
        default="clarify",
        description="The response mode: 'clarify' asks for more info, 'confirm' validates understanding, 'creative' provides imaginative answers, 'technical' gives detailed breakdowns, 'choice' presents options.",
        options=["clarify", "confirm", "creative", "technical", "choice"],
    )
    context: str = Field(
        default="",
        description="Additional context for generating the response.",
    )
    options: list = Field(
        default=[],
        description="For 'choice' mode, the list of options to present to the user.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "What file format should I use for the export?",
                    "mode": "choice",
                    "context": "User is exporting data from a spreadsheet.",
                    "options": ["CSV", "JSON", "Excel"],
                }
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    prompt = kwargs.get("prompt")
    mode = kwargs.get("mode", "clarify")
    context = kwargs.get("context", "")
    options = kwargs.get("options", [])

    if not prompt:
        raise ValueError("A prompt is required for the response.")

    # Build response instructions based on mode
    mode_instructions = {
        "clarify": "Ask clarifying questions to better understand the user's needs. Be specific about what information you need.",
        "confirm": "Confirm your understanding of the request. Summarize what you think the user wants and ask if this is correct.",
        "creative": "Provide a creative and engaging response. Think outside the box while remaining helpful.",
        "technical": "Provide a detailed technical breakdown. Include relevant details, considerations, and step-by-step explanations.",
        "choice": (
            f"Present the following options to the user and ask them to choose: {options}"
            if options
            else "Present relevant options based on the context."
        ),
    }

    instruction = mode_instructions.get(mode, mode_instructions["clarify"])

    # Build the full prompt
    full_prompt = prompt
    if context:
        full_prompt = f"Context:\n{context}\n\nMessage:\n{prompt}"

    return {
        "response_prompt": full_prompt,
        "mode": mode,
        "instruction": instruction,
        "options": options if mode == "choice" else [],
        "status": "ready",
    }
