from pydantic import BaseModel, Field


class Params(BaseModel):
    """Generate up to 4 possible choices for a user to choose from that would satisfy the intent of a given prompt. Each choice offers different trade-offs in terms of optimization, simplicity, flexibility, and scalability."""

    prompt: str = Field(
        ...,
        description="What to generate choices for. Describes the decision or value needed.",
    )
    context: str = Field(
        default="",
        description="Additional context about the workflow, node, or setting being configured.",
    )
    max_choices: int = Field(
        default=4,
        description="Maximum number of choices to generate (2-4).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "What system prompt should be used for an AI node that summarizes meeting notes?",
                    "context": "The workflow processes daily meeting recordings and sends summaries to team leads.",
                    "max_choices": 4,
                }
            ]
        }
    }


async def main(**kwargs) -> dict:
    prompt = kwargs.get("prompt", "")
    context = kwargs.get("context", "")
    max_choices = kwargs.get("max_choices", 4)

    if not prompt:
        raise ValueError("A prompt is required.")

    # Clamp max_choices
    max_choices = max(2, min(4, max_choices))

    letters = ["A", "B", "C", "D"][:max_choices]
    tradeoffs = [
        "emphasizing optimization and efficiency",
        "emphasizing simplicity and ease of use",
        "emphasizing flexibility and adaptability",
        "emphasizing thoroughness and scalability",
    ][:max_choices]

    choice_instructions = "\n".join(
        [f"{letter}: [Choice {tradeoffs[i]}]" for i, letter in enumerate(letters)]
    )

    parts = [
        f"Generate exactly {max_choices} choices for: {prompt}",
    ]

    if context:
        parts.append(f"\nContext: {context}")

    parts.append(
        f"\nEach choice should offer a different trade-off. Mark the most practical one with (RECOMMENDED)."
    )
    parts.append(f"\nRespond in EXACTLY this format:\n{choice_instructions}")

    full_prompt = "\n".join(parts)

    return {
        "choices_prompt": full_prompt,
        "system": "You are a creative assistant that generates distinct, practical choices. Each choice should represent a meaningful trade-off between optimization, simplicity, flexibility, and scalability. Keep choices concise (1-2 sentences each).",
        "max_choices": max_choices,
        "status": "ready",
    }
