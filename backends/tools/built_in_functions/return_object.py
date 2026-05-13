"""Return a JSON object from input data based on a prompt."""

import json
from typing import Dict, Any
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Return a JSON object with key-value pairs from the input data according to the user's instructions."""

    value: Dict[str, Any] = Field(
        ...,
        description="The object with key-value pairs.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": {"name": "John Doe", "age": 30}},
                {"value": {"status": "active", "count": 5, "items": ["a", "b"]}},
                {"value": {}},
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    Return the object.
    The LLM provides the value parameter which is constrained to be a dict.
    """
    value = kwargs.get("value")

    if value is None:
        raise ValueError("value is required")

    # Ensure it's a dict
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot parse '{value}' as JSON object")

    if not isinstance(value, dict):
        raise ValueError(f"Expected an object, got {type(value).__name__}")

    result = {"value": value}
    return result
