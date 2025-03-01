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
def main(args: Params) -> Union[int, float]:
    valueA = args["valueA"]
    valueB = args["valueB"]
    operation = args["operation"]
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
    if not isinstance(valueA, int):
        raise ValueError("valueA must be an int.")
    elif not isinstance(valueB, int):
        raise ValueError("valueB must be an int.")
    elif operation not in possible_operations:
        raise ValueError(
            f"Invalid value {operation} specified for tool parameter 'operation'."
        )

    op_str = possible_operations[operation]
    equation_str = f"{valueA} {op_str} {valueB}"
    result = eval(equation_str)
    return result
