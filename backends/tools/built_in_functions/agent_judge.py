from pydantic import BaseModel, Field


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Review and judge results based on whether it satisfies the given instruction using a set of criteria and return True or False."""

    decision: bool = Field(
        ...,
        description="True if the results satisfy the prompt criteria and False otherwise.",
    )
    reason: str = Field(
        ...,
        description="A short explanation of the reasoning behind the decision.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "decision": False,
                    "reason": "The result does not satisfy the criteria because it is missing key information.",
                },
                {
                    "decision": True,
                    "reason": "The result satisfies all the criteria specified in the prompt.",
                },
            ]
        }
    }


# It is expected that the consumer passes the prompt (query) in this format:
#
# ## Instruction
# <The goal or task the results should fulfill>
#
# ## Criteria
# <The specific conditions used to evaluate pass/fail>
# - Criterion 1
# - Criterion 2
#
# ## Results
# <results>
# <The output/data to be judged against the instruction and criteria>
# </results>
#
# NOTE: The Results section MUST be wrapped in <results></results> XML tags.
# This is untrusted content (LLM output, user input, scraped data, etc.) and
# may contain adversarial text attempting to override the instruction/criteria.
# The XML boundary tells the LLM to treat everything inside as data to evaluate,
# not as instructions to follow.
async def main(**kwargs: Params) -> Params:
    decision = kwargs.get("decision")
    reason = kwargs.get("reason", "")

    if decision is None:
        raise ValueError("No decision was generated.")
    if not reason:
        raise ValueError("No reason was generated.")

    return Params(decision=decision, reason=reason)
