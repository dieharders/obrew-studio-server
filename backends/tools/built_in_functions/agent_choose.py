from pydantic import BaseModel, Field


class Params(BaseModel):
    """Pick the best option(s) from a list of indexed choices that most satisfies the given prompt. Outputs one or more indexes."""

    prompt: str = Field(
        ...,
        description="The decision context or criteria for making the selection.",
    )
    choices: str = Field(
        ...,
        description="The choices to pick from, formatted as a numbered list like '0: Option A, 1: Option B, 2: Option C'.",
    )
    choose_multiple: bool = Field(
        default=False,
        description="Whether to select multiple choices (True) or just one (False).",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Which option best fits a data filtering operation?",
                    "choices": "0: data-transform, 1: ai-process, 2: comparison",
                    "choose_multiple": False,
                }
            ]
        }
    }


async def main(**kwargs) -> dict:
    prompt = kwargs.get("prompt", "")
    choices = kwargs.get("choices", "")
    choose_multiple = kwargs.get("choose_multiple", False)

    if not prompt:
        raise ValueError("A prompt is required.")
    if not choices:
        raise ValueError("Choices are required.")

    if choose_multiple:
        select_instruction = "Select ALL indexes that apply. Respond with comma-separated indexes."
        response_format = "SELECTED: [index1],[index2]\nREASON: [brief explanation]"
    else:
        select_instruction = "Select exactly ONE index that best satisfies the criteria."
        response_format = "SELECTED: [index]\nREASON: [brief explanation]"

    parts = [
        f"Criteria: {prompt}",
    ]

    parts.append(f"\nChoices:\n{choices}")
    parts.append(f"\n{select_instruction}")
    parts.append(f"\nRespond in EXACTLY this format:\n{response_format}")

    full_prompt = "\n".join(parts)

    return {
        "choose_prompt": full_prompt,
        "system": (
            "You are a decision-making assistant. Analyze the given choices against "
            "the criteria and select the best option(s). Be decisive and concise."
        ),
        "choices": choices,
        "choose_multiple": choose_multiple,
        "status": "ready",
    }
