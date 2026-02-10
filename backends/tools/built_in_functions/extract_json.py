"""Extract any JSON value from input data based on a prompt."""

import json
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Extract any valid JSON value from the input data according to the user's instructions."""

    value: str = Field(
        ...,
        description="The extracted JSON value as a JSON-serialized string. For objects: '{\"key\": \"value\"}', for arrays: '[1, 2, 3]', for primitives: '42' or '\"text\"' or 'true' or 'null'.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": '{"key": "value"}'},
                {"value": '[1, 2, 3]'},
                {"value": '"string value"'},
                {"value": '42'},
                {"value": 'true'},
                {"value": 'null'},
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    Return the extracted JSON value.
    The LLM provides the value parameter as a JSON-serialized string.
    """
    value = kwargs.get("value")

    if value is None:
        raise ValueError("value is required")

    # Parse the JSON-serialized string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        # If it's not valid JSON, treat the string itself as the value
        parsed_value = value

    result = {"value": parsed_value}
    return result
