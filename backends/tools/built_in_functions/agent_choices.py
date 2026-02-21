from pydantic import BaseModel, Field


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Given a goal, generate a clarifying question to pose to a user and provide at least 4 options (and a recommended option).\nExample goal: The user wants to create a workflow that processes documents and sends email summaries."""

    question: str = Field(
        ...,
        description="A clarifying question designed to elicit more information from the user about their intended goal.",
    )
    option_0: str = Field(
        ...,
        description="The first option. Must be a short descriptive text the user can read and pick from.",
    )
    option_1: str = Field(
        ...,
        description="The second option. Must be a short descriptive text the user can read and pick from.",
    )
    option_2: str = Field(
        ...,
        description="The third option. Must be a short descriptive text the user can read and pick from.",
    )
    option_3: str = Field(
        ...,
        description="The fourth option. Must be a short descriptive text the user can read and pick from.",
    )
    recommend: str = Field(
        ...,
        description="The field name of the recommended option that best satisfies the intended goal.",
        options=["option_0", "option_1", "option_2", "option_3"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Reading the current graph state, it seems we are missing a node for sending emails. What node should we focus on next?",
                    "option_0": "Email Notify",
                    "option_1": "AI Process",
                    "option_2": "Data Transform",
                    "option_3": "Fetch File",
                    "recommend": "option_0",
                }
            ]
        }
    }


# It is expected that the consumer passes the prompt (query) in this format:
#
# ## Goal
# <The user's intended goal or objective that needs clarification>
#
# ## Context
# <context>
# <Any relevant state, history, or background info to inform the generated options>
# </context>
#
# NOTE: Context is wrapped in <context></context> XML tags because it may contain
# prior LLM output or user-provided text that could be adversarial.
async def main(**kwargs: Params) -> dict:
    question = kwargs.get("question", "")
    recommendation = kwargs.get("recommend", "")

    if not recommendation:
        raise ValueError("No recommendation was generated.")
    if not question:
        raise ValueError("No question was generated.")

    # Assemble options from explicit fields (filter out empty slots)
    options = {}
    for i in range(4):
        value = (kwargs.get(f"option_{i}") or "").strip()
        if value:
            options[str(i)] = value

    if not options:
        raise ValueError("No set of options was generated.")

    # Normalize recommend: "option_0" → "0"
    if recommendation.startswith("option_"):
        recommendation = recommendation.replace("option_", "")

    return {
        "question": question,
        "options": options,
        "recommend": recommendation,
    }
