from typing import Union
from pydantic import BaseModel, Field


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Given a goal, generate a clarifying question to pose to a user and provide multiple choices (up to 4 choices and one recommendation).\nExample goal: The user wants to create a workflow that processes documents and sends email summaries."""

    question: str = Field(
        ...,
        description="A clarifying question designed to elicit more information from the user about their intended goal.",
    )
    choices: dict[str, Union[bool, int, str]] = Field(
        ...,
        description="The choices to pick from that might satisfy the intended goal.",
    )
    recommend: str = Field(
        ...,
        description="The index of the recommended choice that best satisfies the intended goal.",
    )
    # goal: str = Field(
    #     ...,
    #     description="The instruction that needs clarification.",
    #     llm_not_required=True,
    # )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "Reading the current graph state, it seems we are missing a node for sending emails. What node should we focus on next?",
                    # @TODO This may need to be simplified down to `"choice_0": "Email Notify"` like we do in other tools...
                    "choices": {
                        "0": "Email Notify",
                        "1": "AI Process",
                        "2": "Data Transform",
                        "3": "Fetch File",
                    },
                    "recommend": "0",
                    # @TODO This is how the prompt from the frontend should be structured and sent with this tool use
                    # "goal": "The user wants to create a workflow that processes documents and sends email summaries.\n\nWorkflow Rules:\n{...}\n\nCurrent Workflow Graph State:\n'begin'->'file'->'ai_process'\n\nCurrent Plan:\nNew Node = ?\nNode Settings = ?\nNode Connections = ?",
                }
            ]
        }
    }


# The FrontEnd should use the `recommend` prop to add text notation the agent's choices:
#
# QUESTION: [Clarifying question here]
# A: [First choice] (RECOMMENDED)
# B: [Second choice]
# C: [Third choice]
# D: [Fourth choice]
async def main(**kwargs: Params) -> dict:
    question = kwargs.get("question", "")
    choices = kwargs.get("choices", dict())
    recommendation = kwargs.get("recommend", "")

    if not recommendation:
        raise ValueError("No recommendation was generated.")
    if not question:
        raise ValueError("No question was generated.")
    if not choices:
        raise ValueError("No set of choices was generated.")

    return {
        "question": question,
        "choices": choices,
        "recommend": recommendation,
    }
