from pydantic import BaseModel, Field


class Params(BaseModel):
    """Evaluate and write a multi-step plan. Given a goal and the current plan, determine if the plan has enough information to complete the goal, then write or update the plan."""

    goal: str = Field(
        ...,
        description="The goal to accomplish.",
    )
    current_plan: str = Field(
        default="",
        description="The current state of the plan with accumulated decisions and context.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "goal": "Create a workflow that processes documents and sends email summaries",
                    "current_plan": "Purpose: Document processing\nFrequency: Daily\nInput: PDF files from shared drive",
                }
            ]
        }
    }


RESPONSE_FORMAT = """Evaluate the current plan against the goal. Then respond in EXACTLY one of these formats:

If the plan has ENOUGH information to proceed, respond with:
READY
[Your final multi-step plan here]

If MORE information is needed, respond with:
CONTINUE
[Your updated plan with what is known so far and what still needs to be determined]"""


async def main(**kwargs) -> dict:
    goal = kwargs.get("goal", "")
    current_plan = kwargs.get("current_plan", "")

    if not goal:
        raise ValueError("A goal is required.")

    parts = [f"Goal: {goal}"]

    if current_plan:
        parts.append(f"\nCurrent plan:\n{current_plan}")

    parts.append(f"\n{RESPONSE_FORMAT}")

    full_prompt = "\n".join(parts)

    return {
        "plan_prompt": full_prompt,
        "system": (
            "You are a planning assistant. Evaluate whether the current plan has enough "
            "information to accomplish the stated goal. If yes, output READY followed by the "
            "finalized plan. If no, output CONTINUE followed by an updated plan noting what "
            "is known and what still needs to be determined. Be concise and actionable."
        ),
        "goal": goal,
        "status": "ready",
    }
