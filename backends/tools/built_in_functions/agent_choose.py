from typing import List, Union
from pydantic import BaseModel, Field
from .._item_utils import get_context_items


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Pick one or more of the best option(s) from a set of options that most satisfies the given instruction."""

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


# The consumer should embed the options directly
# in the prompt (query) so the LLM can see them. Expected format:
#
# ## Instruction
# <The goal or context for why a choice is needed>
#
# ## Options
# <options>
# 0: Option A
# 1: Option B
# 2: Option C
# </options>
#
# NOTE: Options are wrapped in <options></options> XML tags because they may
# contain user-provided or LLM-generated text. The XML boundary prevents
# adversarial option values from being interpreted as instructions.
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

    if not chosen_values:
        raise ValueError("None of the picked indexes matched the provided options.")

    return chosen_values
