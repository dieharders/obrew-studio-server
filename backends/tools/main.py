import os
import importlib.util
from typing import Type
from core.classes import ToolFunctionSchema
from core import common
from pydantic import BaseModel


# def convert_type_strings(inStr: str):
#     match inStr:
#         case "List" | "Enum":
#             return "array"
#         case "float" | "int":
#             return "number"
#         case "str":
#             return "string"
#         case "dict" | "Dict":
#             return "object"
#     return inStr


# Load the code module and pydantic model for the tool
def load_function_file(filename: str) -> ToolFunctionSchema:
    spec = None

    # Check built-in funcs first
    try:
        # from _deps directory
        prebuilt_funcs_path = common.dep_path(
            os.path.join(
                "backends", common.TOOL_FOLDER, common.TOOL_FUNCS_FOLDER, filename
            )
        )
        if os.path.exists(prebuilt_funcs_path):
            spec = importlib.util.spec_from_file_location(
                name=filename,
                location=prebuilt_funcs_path,
            )
    except Exception as err:
        print(f"{common.PRNT_API} {err}")

    # Check user made funcs
    try:
        # from root of installation dir
        custom_funcs_path = os.path.join(os.getcwd(), common.TOOL_FUNCS_PATH, filename)
        if os.path.exists(custom_funcs_path):
            spec = importlib.util.spec_from_file_location(
                name=filename,
                location=custom_funcs_path,
            )
    except Exception as err:
        print(f"{common.PRNT_API} {err}")

    if not spec:
        raise Exception("No tool found.")

    try:
        tool_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool_code)
        # tool_func = getattr(tool_code, "main") # used to get the function itself
        pydantic_model_name = "Params"
        pydantic_model: Type[BaseModel] = getattr(tool_code, pydantic_model_name)
        schema = pydantic_model.model_json_schema()
        tool_description = schema.get("description", "This is a tool.")
        examples = schema.get("examples", [dict()])
        example_tool_schema: dict = None
        if len(examples) > 0:
            example_tool_schema = examples[0]
        properties: dict = schema.get("properties", dict)
        # Make params
        params = []
        tool_schema = dict()
        for prop in properties.items():
            key = prop[0]
            param = dict(**prop[1])
            param["name"] = key
            # Create tool schema
            allowed_values = param.get("options", None)
            schema_params = dict(
                description=param.get("description", ""),
                type=param.get("type", ""),
            )
            if allowed_values:
                schema_params["allowed_values"] = allowed_values
            tool_schema[key] = schema_params
            # Record param
            params.append(param)
        return {
            "description": tool_description,
            "params": params,
            # Schema for params to pass for tool use
            "params_schema": tool_schema,
            # All required params with example values
            "params_example": example_tool_schema,
        }
    except Exception as err:
        print(f"{common.PRNT_API} Error loading tool function: {err}")
        raise err


# @TODO Implement
def use_tool(args: dict):
    schema: dict = args.get("schema")
    param = dict()
    llm_params = dict()
    llm_tool_example = dict()
    key = param.get("name")
    example_schema = schema.get("examples", [dict()])[0]
    # Make llm response schema (if required)
    allowed_values = param.get("options", None)
    llm_schema_params = dict(
        description=param.get("description", ""),
        type=param.get("type", ""),
    )
    if allowed_values:
        llm_schema_params["allowed_values"] = allowed_values
    llm_tool_example[key] = llm_schema_params
    # Figure out what props are needed from llm
    if not param.get("input_type") and not param.get("value"):
        if key in example_schema:
            llm_params[key] = example_schema[key]
    # Return schema for llm to respond with
    return llm_params
