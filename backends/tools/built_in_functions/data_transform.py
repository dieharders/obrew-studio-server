import json
from typing import Literal, Any
from pydantic import BaseModel, Field


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Transform and restructure data according to specified rules."""

    transformed_data: str = Field(
        ...,
        description="Only the resulting value of the transformed data.",
    )
    data_type: Literal[
        "string", "number", "boolean", "xml", "json", "array", "date", "null"
    ] = Field(
        ...,
        description="The data type of the transformed data value.",
        options=["string", "number", "boolean", "xml", "json", "array", "date", "null"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "transformed_data": {
                        "name": "John Doe",
                        "email": "john@example.com",
                    },
                    "data_type": "json",
                },
                {
                    "transformed_data": [100, 200, 300],
                    "data_type": "array",
                },
                {
                    "transformed_data": "Transformed text result",
                    "data_type": "string",
                },
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    transformed_data = kwargs.get("transformed_data")
    data_type = kwargs.get("data_type")

    # Checks
    if transformed_data is None:
        raise ValueError("transformed_data is required.")

    if not data_type:
        raise ValueError("data_type is required.")

    # Parse the data according to the data_type
    parsed_data = transformed_data

    if data_type == "string":
        parsed_data = str(transformed_data)
    elif data_type == "number":
        # Try to parse as float or int
        try:
            parsed_data = (
                float(transformed_data)
                if "." in str(transformed_data)
                else int(transformed_data)
            )
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert '{transformed_data}' to number.")
    elif data_type == "boolean":
        if isinstance(transformed_data, bool):
            parsed_data = transformed_data
        elif isinstance(transformed_data, str):
            parsed_data = transformed_data.lower() in ("true", "1", "yes")
        else:
            parsed_data = bool(transformed_data)
    elif data_type == "json":
        # If it's a string, try to parse it as JSON
        if isinstance(transformed_data, str):
            try:
                parsed_data = json.loads(transformed_data)
            except json.JSONDecodeError:
                raise ValueError(f"Cannot parse '{transformed_data}' as JSON.")
        else:
            # Already a dict/list, keep as is
            parsed_data = transformed_data
    elif data_type == "array":
        if isinstance(transformed_data, list):
            parsed_data = transformed_data
        elif isinstance(transformed_data, str):
            try:
                parsed_data = json.loads(transformed_data)
                if not isinstance(parsed_data, list):
                    raise ValueError("Parsed data is not an array.")
            except json.JSONDecodeError:
                raise ValueError(f"Cannot parse '{transformed_data}' as array.")
        else:
            raise ValueError(f"Cannot convert '{transformed_data}' to array.")
    elif data_type == "xml":
        # XML is kept as string
        parsed_data = str(transformed_data)
    elif data_type == "date":
        # Date is kept as string (ISO format expected)
        parsed_data = str(transformed_data)
    elif data_type == "null":
        parsed_data = None

    # Set result structure
    result = {
        "data": parsed_data,
        "data_type": data_type,
    }

    # Return JSON string
    return result
