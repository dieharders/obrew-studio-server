"""Return a string value from input data based on a prompt."""

from pydantic import BaseModel, Field


class Params(BaseModel):
    """Return a text/string value from the input data according to the user's instructions."""

    value: str = Field(
        ...,
        description="The string value. Return only the text content requested.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": "John Doe"},
                {"value": "The quick brown fox jumps over the lazy dog"},
                {"value": "user@example.com"},
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    Return the string value.
    The LLM provides the value parameter which is already constrained to be a string.
    """
    value = kwargs.get("value")

    if value is None:
        raise ValueError("value is required")

    result = {"value": str(value)}
    return result
