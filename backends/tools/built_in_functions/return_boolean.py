from pydantic import BaseModel, Field


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Return a boolean (True/False) value. Use this for binary YES/NO decisions, binary comparisons, or branch logic based on the user's instructions."""

    value: bool = Field(
        ...,
        description="True or False based on evaluating the input against the user's instruction.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"value": True},
                {"value": False},
            ]
        }
    }


async def main(**kwargs) -> dict:
    """
    This is a simple, general purpose tool. The LLM provides a generic `value` parameter constrained to a boolean.
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
    return result
