from typing import List, Union
from pydantic import BaseModel, Field


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Pick one or more of the best option(s) from a set of choices that most satisfies the given prompt."""

    # We make the LLM to output this so it forces it to choose whether to pick one or multiple options.
    choose_multiple: bool = Field(
        default=False,
        description="Whether to select multiple choices (True) or just one (False).",
    )
    pick: List[str] = Field(
        ...,
        description="A list of index(s) of the chosen option(s).",
    )
    # Provided by FrontEnd
    choices: dict[str, Union[bool, int, str]] = Field(
        ...,
        description="The choices to pick from, formatted as an indexed set of key/value pairs.",
        llm_not_required=True,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "choose_multiple": False,
                    "choices": {"0": "Option A", "1": "Option B", "2": "Option C"},
                    "pick": ["1"],
                },
                {
                    "choose_multiple": True,
                    "choices": {"0": 32, "1": 64, "2": 128},
                    "pick": ["0", "2"],
                },
            ]
        }
    }


async def main(**kwargs: Params) -> List[Union[bool, int, str]]:
    chosen_indexes = kwargs.get("pick", [])
    choices = kwargs.get("choices", dict())

    if not chosen_indexes:
        raise ValueError("No pick was generated.")
    if not choices:
        raise ValueError("No choices are provided.")

    # Get the chosen values from the choices provided
    chosen_values = [choices[str(i)] for i in chosen_indexes if str(i) in choices]

    return chosen_values
