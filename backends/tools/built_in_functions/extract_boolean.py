"""Extract a boolean value from input data based on a prompt."""

import json
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Extract a boolean (true/false) value from the input data according to the user's instructions."""

    value: bool = Field(
        ...,
        description="The extracted boolean value. Return true or false based on the analysis.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": True},
                {"value": False},
            ]
        }
    }


async def main(**kwargs) -> str:
    """
    Return the extracted boolean value.
    The LLM provides the value parameter which is constrained to be a boolean.
    """
    value = kwargs.get("value")

    if value is None:
        raise ValueError("value is required")

    # Ensure it's a boolean
    if isinstance(value, str):
        value = value.lower() in ("true", "1", "yes")
    else:
        value = bool(value)

    result = {"value": value}
    return json.dumps(result, indent=2)
