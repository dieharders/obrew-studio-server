from pydantic import BaseModel, Field


class Params(BaseModel):
    """Generate a clarifying question with multiple choices. Given an instruction and current plan context, return a follow-up question to provide task clarity along with up to 4 choices and one recommendation."""

    instruction: str = Field(
        ...,
        description="The instruction or goal that needs clarification.",
    )
    current_plan: str = Field(
        default="",
        description="The current plan context with accumulated decisions so far.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instruction": "Create a workflow that processes documents and sends email summaries",
                    "current_plan": "Purpose: Document processing\nFrequency: Daily",
                }
            ]
        }
    }


RESPONSE_FORMAT = """Respond in EXACTLY this format:
QUESTION: [Your clarifying question here]
A: [First choice] (RECOMMENDED)
B: [Second choice]
C: [Third choice]
D: [Fourth choice]"""


async def main(**kwargs) -> dict:
    instruction = kwargs.get("instruction", "")
    current_plan = kwargs.get("current_plan", "")

    if not instruction:
        raise ValueError("An instruction is required.")

    parts = [f"Goal: {instruction}"]

    if current_plan:
        parts.append(f"\nCurrent plan:\n{current_plan}")

    parts.append(
        "\nAsk ONE clarifying question with up to 4 choices that would help "
        "refine the plan. Each choice should offer a meaningfully different approach. "
        "Mark your recommended choice with (RECOMMENDED)."
    )
    parts.append(f"\n{RESPONSE_FORMAT}")

    full_prompt = "\n".join(parts)

    return {
        "choices_prompt": full_prompt,
        "system": (
            "You are an assistant that asks clarifying questions to help accomplish a task. "
            "Generate exactly one question at a time with up to 4 practical, distinct choices. "
            "Mark the most practical choice with (RECOMMENDED). Be concise."
        ),
        "instruction": instruction,
        "status": "ready",
    }
