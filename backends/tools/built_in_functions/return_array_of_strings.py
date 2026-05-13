"""Return an array of strings from input data based on a prompt."""

import json
from typing import List
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Return a list of string values from the input data according to the user's instructions."""

    value: List[str] = Field(
        ...,
        description="The array of strings. Each item should be a separate string element.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": ["apple", "banana", "cherry"]},
                {"value": ["user1@example.com", "user2@example.com"]},
                {"value": ["Item 1", "Item 2", "Item 3"]},
                {"value": []},
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    Return the array of strings.
    The LLM provides the value parameter which is constrained to be a list of strings.
    """
    value = kwargs.get("value")

    if value is None:
        raise ValueError("value is required")

    # Ensure it's a list
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            # Try splitting by comma as fallback
            value = [item.strip() for item in value.split(",")]

    if not isinstance(value, list):
        raise ValueError(f"Expected a list, got {type(value).__name__}")

    # Ensure all items are strings
    value = [str(item) for item in value]

    result = {"value": value}
    return result
