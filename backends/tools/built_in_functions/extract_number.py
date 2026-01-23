"""Extract a numeric value from input data based on a prompt."""

import json
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Extract a numeric value from the input data according to the user's instructions."""

    value: float = Field(
        ...,
        description="The extracted numeric value. Return only the number, not text like '5 items'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": 42},
                {"value": 3.14159},
                {"value": -100},
                {"value": 0},
            ]
        }
    }


async def main(**kwargs) -> str:
    """
    Return the extracted numeric value.
    The LLM provides the value parameter which is constrained to be a number.
    """
    value = kwargs.get("value")

    if value is None:
        raise ValueError("value is required")

    # Ensure it's a number
    if isinstance(value, str):
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            raise ValueError(f"Cannot convert '{value}' to a number")

    result = {"value": value}
    return json.dumps(result, indent=2)
