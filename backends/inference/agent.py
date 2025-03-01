import json
import os
from typing import Any, List
import re
import json
from pydantic import BaseModel
from core import classes

# import importlib.util
# from core import common


# Load the code module and pydantic model for the tool
# file name and function name must be the same!
# def load_function_file(filename: str):
#     func_name = os.path.splitext(filename)[0]
#     prebuilt_funcs_path = common.dep_path(os.path.join(common.TOOL_PREBUILT_PATH, filename))
#     custom_funcs_path = os.path.join(common.TOOL_FUNCS_PATH, filename)
#     try:
#         # Check pre-made funcs first
#         spec = importlib.util.spec_from_file_location(
#             name=filename,
#             location=prebuilt_funcs_path,
#         )
#     except:
#         pass
#     try:
#         # Check custom user funcs
#         spec = importlib.util.spec_from_file_location(
#             name=filename,
#             location=custom_funcs_path,
#         )
#     except:
#         raise Exception("No path/function found.")
#     if not spec:
#         raise Exception("No tool found.")
#     tool_code = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(tool_code)
#     tool_func = getattr(tool_code, func_name)
#     pydantic_model_name = "Params"
#     pydantic_model = getattr(tool_code, pydantic_model_name)
#     return {
#         "func": tool_func,
#         "model": pydantic_model.model_json_schema(),
#         # "model": pydantic_model.__annotations__,
#     }


# Return arguments in a (Pydantic) schema and example output
def construct_arguments(schema: Any):
    args: dict[str, dict[str, str]] = schema["properties"]
    example_args = schema["examples"][0]  # We only take 1st example
    required_args = schema["required"]

    # Function to extract the field types and names from the Pydantic model
    def get_model_fields() -> dict:
        fields = {}
        for name, field in args.items():
            is_required = name in required_args
            data = {
                "type": field["type"],
                "required_arg": is_required,
            }
            if "enum" in field:
                data["allowed_values"] = field["enum"]
            fields[name] = data
        return fields

    # Get the fields
    fields = get_model_fields()

    # Return a json for example and args
    return {
        "arguments": fields,
        "required_arguments": required_args,
        "example_arguments": example_args,
    }


def dict_list_to_markdown(dict_list: List[dict]):
    markdown_string = ""
    for index, item in enumerate(dict_list):
        markdown_string += f"# Tool {index + 1}: {item.get("name")}\n\n"
        for key, value in item.items():
            # If code
            if key == "arguments" or key == "example_arguments":
                markdown_string += f"## {key}\n```json\n{value}\n```\n\n"
            else:
                markdown_string += f"## {key}\n{value}\n\n"
    return markdown_string


# Filter out all "required_arg" props
def get_allowed_args(tool_args: dict):
    result = []
    for name, value in tool_args.items():
        if value.get("required_arg") == True:
            result.append(name)
    return result


def get_tool_props(tool_def: classes.ToolDefinition):
    tool_descr_str = tool_def["description"]
    tool_name_str = tool_def["name"]
    # Build components of prompt
    tool_args_json = json.dumps(tool_def["arguments"], indent=4)
    tool_args_str = f"```json\n{tool_args_json}\n```"
    tool_example_json = json.dumps(tool_def["example_arguments"], indent=4)
    tool_example = f"```json\n{tool_example_json}\n```"
    tool_allowed_keys = get_allowed_args(tool_def["arguments"])
    return {
        "name": tool_name_str,
        "description": tool_descr_str,
        "arguments": tool_args_str,
        "example_arguments": tool_example,
        "allowed_arguments": tool_allowed_keys,
    }


# Get the actual function using the provided path
# def get_tool_function(path: str, tool: classes.ToolDefinition):
#     is_url = path.startswith("https://")
#     tool_code = load_function_file(filename=tool["path"])
#     tool_func = tool_code["func"]
#     # @TODO Construct api call for endpoint
#     if is_url:
#         return tool_func
#     # Load local func
#     else:
#         return tool_func


# Pass the text response which includes the function params
# def eval(tool: classes.ToolDefinition, args: dict):
#     # pass llm response to function
#     path = tool["path"]
#     tool_function = get_tool_function(tool=tool, path=path)
#     return tool_function(args)


class ParsedOutput(BaseModel):
    raw: str
    text: str


# Parse out the json result using either regex or another llm call
def parse_output(output: str, tool_def: classes.ToolDefinition) -> ParsedOutput:
    print(f"Agent output response::\n{output}")

    tool_attrs = get_tool_props(tool_def=tool_def)
    allowed_arguments: List[str] = tool_attrs.get("allowed_arguments", [])
    parsed_output = {
        "raw": "",
        "text": "",
    }
    pattern_object = r"({.*?})"
    pattern_json_object = r"\`\`\`json\n({.*?})\n\`\`\`"
    match_json_object = re.search(pattern_json_object, output, re.DOTALL)
    match_object = re.search(pattern_object, output, re.DOTALL)

    if match_json_object or match_object:
        # Find first occurance
        if match_json_object:
            json_block = match_json_object.group(1)
        elif match_object:
            json_block = match_object.group(1)
        # Remove single-line comments (//...)
        json_block = re.sub(r"//.*", "", json_block)
        # Remove multi-line comments (/*...*/)
        json_block = re.sub(r"/\*.*?\*/", "", json_block, flags=re.DOTALL)
        # Clean up any extra commas or trailing whitespace
        json_block = re.sub(r",\s*(\}|\])", r"\1", json_block)
        json_block = json_block.strip()
        # Convert JSON block back to a dictionary to ensure it's valid JSON
        try:
            # Remove any unrelated keys from json
            json_object: dict = json.loads(json_block)
            # Filter out keys not in the allowed_keys set
            filtered_json_object = {
                k: v for k, v in json_object.items() if k in allowed_arguments
            }
            result = eval(
                tool=tool_def,
                args=filtered_json_object,
            )
            # Preserve the correct type
            parsed_output["raw"] = {"result": result}
            parsed_output["text"] = f"{result}"
            print(f"Agent answer:: {parsed_output}")
            return parsed_output
        except json.JSONDecodeError as e:
            print("Invalid JSON:", e)
            raise Exception("Invalid JSON.")
    else:
        raise Exception("No JSON block found!")


# Create arguments and example response for llm prompt from pydantic model
# def create_tool_args(tool_def: classes.ToolSaveRequest) -> classes.ToolDefinition:
#     new_dict = dict(arguments={}, example_arguments={}, description="")
#     # Get values
#     new_def = {**new_dict, **tool_def.model_dump()}
#     tool_code = load_function_file(filename=tool_def.path)
#     tool_model = tool_code["model"]
#     tool_schema = construct_arguments(tool_model)
#     tool_description = tool_model["description"]
#     tool_args = tool_schema.get("arguments", {})
#     tool_example_args = tool_schema.get("example_arguments", {})
#     # Set empty attrs
#     new_def["arguments"] = tool_args
#     new_def["example_arguments"] = tool_example_args
#     new_def["description"] = tool_description or "This is a tool."
#     return new_def
