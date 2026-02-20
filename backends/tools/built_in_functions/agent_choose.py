from typing import List, Union
from pydantic import BaseModel, Field
from .._item_utils import get_context_items


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Pick one or more of the best option(s) from a set of options that most satisfies the given prompt."""

    # We make the LLM to output this so it forces it to choose whether to pick one or multiple options.
    choose_multiple: bool = Field(
        default=False,
        description="Whether to select multiple options (True) or just one (False).",
    )
    pick: List[str] = Field(
        ...,
        description="A list of index key(s) of the chosen option(s).",
    )
    # Provided by FrontEnd
    options: dict[str, Union[bool, int, str]] = Field(
        ...,
        description="The options to pick from, formatted as an indexed set of key/value pairs.",
        llm_not_required=True,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "choose_multiple": False,
                    "options": {"0": "Option A", "1": "Option B", "2": "Option C"},
                    "pick": ["1"],
                },
                {
                    "choose_multiple": True,
                    "options": {"0": 32, "1": 64, "2": 128},
                    "pick": ["0", "2"],
                },
            ]
        }
    }


# @TODO This is apparently broken. The model cannot respond with a "pick" in response.
async def main(**kwargs: Params) -> List[Union[bool, int, str]]:
    chosen_indexes = kwargs.get("pick", [])
    options = kwargs.get("options", dict())

    # Fallback: read options from context_items (passed by frontend)
    if not options:
        items = get_context_items(kwargs)
        for item in items:
            if "options" in item:
                options = item["options"]
                break

    if not chosen_indexes:
        raise ValueError("No pick was generated.")
    if not options:
        raise ValueError("No options are provided.")

    # Get the chosen values from the options provided
    chosen_values = [options[str(i)] for i in chosen_indexes if str(i) in options]

    return chosen_values
