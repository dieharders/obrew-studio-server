from typing import Literal, Union
from pydantic import BaseModel, Field


# Required - Always use "Params" as Pydantic model name
class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Perform simple arithmetic on numbers (operands) according to the specified operation."""
    value_a: int = Field(
        ...,
        description="The first numerical value for the operation.",
    )
    value_b: int = Field(
        ...,
        description="The second numerical value for the operation.",
    )
    operation: Literal["multiply", "add", "subtract", "divide"] = Field(
        ...,
        description="The mathematical operation to perform.",
        options=["multiply", "add", "subtract", "divide"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "value_a": 2,
                    "value_b": 6,
                    "operation": "add",
                }
            ]
        }
    }


# Required - Put custom code in folder [root]/tools/functions/
# Or
# Required - Put built-in code in folder backends/tools/built-in/
#
# Required - Functions must be asynchronous regardless
async def main(**kwargs: Params) -> Union[int, float]:
    value_a = kwargs["value_a"]
    value_b = kwargs["value_b"]
    operation = kwargs["operation"]
    # Dont need all these this since we inform llm of what the allowed values are,
    # but better safe than sorry.
    possible_operations = {
        "add": "+",
        "+": "+",
        "addition": "+",
        "-": "-",
        "subtract": "-",
        "subtraction": "-",
        "*": "*",
        "mul": "*",
        "multiple": "*",
        "x": "*",
        "X": "*",
        "multiplication": "*",
        "multiply": "*",
        "/": "/",
        "divide": "/",
        "div": "/",
        "division": "/",
    }
    if not isinstance(value_a, int):
        raise ValueError("value_a must be an int.")
    elif not isinstance(value_b, int):
        raise ValueError("value_b must be an int.")
    elif operation not in possible_operations:
        raise ValueError(
            f"Invalid value {operation} specified for tool parameter 'operation'."
        )

    op_str = possible_operations[operation]
    equation_str = f"{value_a} {op_str} {value_b}"
    result = eval(equation_str)
    return result
