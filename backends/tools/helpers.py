import os
import re
import json
import importlib.util
from typing import List, Optional
from core import common
from core.classes import ToolDefinition, ToolFunctionParameter

TOOL_FUNCTION_NAME = "main"


def import_tool_function(filename: str):
    spec = None

    # Check built-in funcs first
    try:
        # from _deps directory
        prebuilt_funcs_path = common.dep_path(
            os.path.join(
                common.BACKENDS_FOLDER,
                common.TOOL_FOLDER,
                common.TOOL_FUNCS_FOLDER,
                filename,
            )
        )
        if os.path.exists(prebuilt_funcs_path):
            spec = importlib.util.spec_from_file_location(
                name=filename,
                location=prebuilt_funcs_path,
            )
    except Exception as err:
        print(f"{common.PRNT_API} {err}", flush=True)

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
        print(f"{common.PRNT_API} {err}", flush=True)

    if not spec:
        raise Exception("No tool found.")

    return spec


def load_function(filename: str):
    try:
        spec = import_tool_function(filename)
        tool_code = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool_code)
        func = getattr(tool_code, TOOL_FUNCTION_NAME)
        return func
    except Exception as err:
        print(f"{common.PRNT_API} {err}", flush=True)
        raise Exception("Failed to load tool function.")


# Good for explaining each tool to llm so it can make a selection.
def tool_to_markdown(tool_list: List[ToolDefinition], include_code=False):
    """Create a machine readable description of a tool definition."""
    markdown_string = ""
    for index, item in enumerate(tool_list):
        markdown_string += f"## Tool {index + 1}\n\n"
        for key, value in item.items():
            # If code
            if key == "params" or key == "params_example" or key == "params_schema":
                if include_code:
                    markdown_string += (
                        f"### {key}\n```json\n{json.dumps(value, indent=4)}\n```\n\n"
                    )
            # Add descr
            elif key == "description" or key == "name":
                markdown_string += f"### {key}\n\n{value}\n\n"
            # Add tool return type
            elif key == "output_type" and include_code:
                return_type = ", ".join(value)
                if return_type:
                    markdown_string += f"### Return type:\n\n{return_type}"
    return markdown_string


# Parse out the json from llm tool call response using either regex or another llm call
def parse_structured_llm_response(
    arguments_str: str, allowed_arguments: List[str]
) -> dict:
    pattern_object = r"({.*?})"
    pattern_json_object = r"\`\`\`json\n({.*?})\n\`\`\`"
    match_json_object = re.search(pattern_json_object, arguments_str, re.DOTALL)
    match_object = re.search(pattern_object, arguments_str, re.DOTALL)

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
            return filtered_json_object
        except json.JSONDecodeError as e:
            raise Exception("Invalid JSON.")
    else:
        raise Exception("No JSON block found!")


# Conversion func to translate our tool def to a native tool (openai compatible) json schema.
# Most models will likely use this, but some, like Functionary use other formats.
def tool_to_json_schema(name: str, description: str, params: List[dict]) -> str:
    # Construct parameters
    properties = {}
    required = []
    for param in params:
        param_name = param["name"]
        param_type = param["type"]
        param_description = param["description"]
        param_schema = {"type": param_type, "description": param_description}
        properties[param_name] = param_schema
        required.append(param_name)
    # Construct new schema
    # @TODO Add output type
    native_schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",  # For now, props all passed as object
                "properties": properties,
                "required": required,
            },
        },
    }
    return json.dumps(native_schema)


# Conversion func to translate our tool def to a native tool (typescript) schema
def tool_to_typescript_schema(name: str, description: str, params: List[dict]) -> str:
    # Construct parameters
    properties = ""
    output_type = "any"  # @TODO Define output type
    for param in params:
        param_name = param["name"]
        param_type = param["type"]
        param_description = param["description"]
        param_str = f"// {param_description}\n{param_name}: {param_type},"
        properties += f"\n{param_str}"
    # Construct new schema
    native_schema = f"""// {description}
type {name} = (_: {{
{properties}
}}) => {output_type};"""
    # Result
    return native_schema


# Convert a tool's schema to markdown text
def schema_to_markdown(schema: dict) -> str:
    result = ""
    for index, pname in enumerate(schema):
        descr = schema[pname].get("description", None)
        data_type = schema[pname].get("type", None)
        allowed_values = schema[pname].get("allowed_values", None)
        if index > 0:
            result += "\n\n"
        result += f"### {pname}\n\nDescription: {descr}\nData type: {data_type}"
        if allowed_values:
            result += f"\nAllowed values: {allowed_values}"
    return result


def get_required_examples(required: List[str], example: dict) -> dict:
    result = dict()
    for pname in required:
        if pname in example:
            result[pname] = example[pname]
    return result


def get_required_schema(required: List[str], schema: dict) -> dict:
    result = dict()
    for pname in required:
        if pname in schema:
            result[pname] = schema[pname]
    return result


def get_provided_args(args_str: str, tool_params: dict):
    provided_arguments = dict()
    for pname in tool_params:
        value = tool_params[pname].get("value", None)
        if pname == "prompt":
            provided_arguments[pname] = args_str
            continue
        if value:
            provided_arguments[pname] = value
    return provided_arguments


# Determine allowed arg names (arguments that llm needs to fill in)
def get_llm_required_args(tool_params: List[ToolFunctionParameter]) -> List[str]:
    result = []
    # Check each `value` exists, if not then the llm needs to send it,
    # we ignore "prompt" since its always provided.
    for param in tool_params:
        pname = param.get("name", None)
        if pname and pname != "prompt" and not param.get("value", None):
            result.append(pname)
    return result


def find_tool_in_response(response: str, tools: List[str]) -> Optional[str]:
    """Attempt to find any tool names in the given response text."""
    print(f"{common.PRNT_API} Searching in response for tools...{tools}", flush=True)
    for tool_name in tools:
        match = re.search(re.escape(tool_name), response, re.DOTALL)
        if match:
            print(f"{common.PRNT_API} Found tool:{tool_name}", flush=True)
            return tool_name  # Return first match immediately
    return None  # Explicitly return None if no tool is found
