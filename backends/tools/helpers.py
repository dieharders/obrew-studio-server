import os
import re
import json
import importlib.util
from typing import List, Optional
from core import common
from core.classes import ToolDefinition, ToolFunctionParameter
from json_repair import repair_json

TOOL_FUNCTION_NAME = "main"
# Structured output schemas (json)
KEY_TOOL_NAME = "tool_choice"
KEY_TOOL_PARAMS = "tool_parameters"
TOOL_CHOICE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        KEY_TOOL_NAME: {"type": "string"},
    },
    "required": [KEY_TOOL_NAME],
}
TOOL_CHOICE_SCHEMA = {KEY_TOOL_NAME: "name"}
TOOL_CHOICE_SCHEMA_STR = f"```json\n{json.dumps(TOOL_CHOICE_SCHEMA)}\n```"
TOOL_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        KEY_TOOL_NAME: {"type": "string"},
        KEY_TOOL_PARAMS: {"type": "object", "properties": {}, "required": []},
    },
    "required": [KEY_TOOL_NAME, KEY_TOOL_PARAMS],
}
TOOL_OUTPUT_SCHEMA = {KEY_TOOL_NAME: "str", KEY_TOOL_PARAMS: "dict"}
TOOL_OUTPUT_SCHEMA_STR = f"```json\n{json.dumps(TOOL_OUTPUT_SCHEMA)}\n```"


def import_tool_function(filename: str):
    spec = None

    # Strip .py extension if present to get the module name
    module_name_only = (
        filename.replace(".py", "") if filename.endswith(".py") else filename
    )

    # Check in base dev path for built_in_functions
    try:
        module_name = f"tools.built_in_functions.{module_name_only}"
        spec = importlib.util.find_spec(module_name)
        if spec:
            print(
                f"{common.PRNT_API} Found spec in built_in_functions: {module_name}",
                flush=True,
            )
    except Exception as err:
        print(
            f"{common.PRNT_API} Error finding spec in built_in_functions: {err}",
            flush=True,
        )

    # Check in Production path /tools/functions for user added tool funcs
    try:
        base_path = common.app_path()
        custom_funcs_path = os.path.join(
            base_path, common.TOOL_FOLDER, common.TOOL_FUNCS_FOLDER, filename
        )
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


# Filter out keys not in the allowed_keys set
def filter_allowed_keys(schema: dict, allowed: List[str] = []):
    filtered_json_object = None
    # Filter out keys not in the allowed_keys set
    if len(allowed) > 0:
        filtered_json_object = {k: v for k, v in schema.items() if k in allowed}
    return filtered_json_object


def strip_extra_chars(text: str):
    result = text.strip()
    # Remove single-line comments (//...)
    result = re.sub(r"//.*", "", text)
    # Remove multi-line comments (/*...*/)
    result = re.sub(r"/\*.*?\*/", "", result, flags=re.DOTALL)
    # Clean up any extra commas or trailing whitespace
    result = re.sub(r",\s*(\}|\])", r"\1", result)
    result = result.strip()
    return result


# Parse out the json from llm tool call response using either regex or another llm call
def parse_json_block(text: str):
    json_str = text

    # Try the raw response
    try:
        # Check if response starts/ends with {}
        t = text.strip()
        is_object = t.startswith("{") and t.endswith("}")
        if is_object:
            result_dict = repair_json(json_str, return_objects=True)
            return result_dict
    except:
        pass

    # Find all matches - try to extract JSON from markdown code blocks
    pattern_json = r"```json\s*\r?\n([\s\S]*?)\r?\n\s*```"
    pattern_ticks = r"```\s*([\s\S]*?)\s*```"
    patterns = [pattern_json, pattern_ticks]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            break

    # If none of the above patterns worked, try a more aggressive approach
    if json_str == text:
        json_str = re.sub(
            r"`{1,3}\s*json\s*\r?\n(.*?)`{1,3}", r"\1", text, flags=re.DOTALL
        )

    # Fail, send response back (llm may have reason for missing result)
    if json_str == text:
        print(f"{common.PRNT_API} No JSON block found.", flush=True)
        return None

    try:
        # Clean up
        json_str = strip_extra_chars(json_str)
        # Convert JSON block back to a dictionary to ensure it's valid JSON
        # https://github.com/mangiucugna/json_repair
        json_object = repair_json(json_str, return_objects=True)
        return json_object
    except json.JSONDecodeError as e:
        raise Exception("Invalid JSON.")


# Parse out the json from llm tool call response using regex
def parse_tool_response(json_str: str, allowed_arguments: List[str] = []):
    json_dict = parse_json_block(json_str)
    if not json_dict:
        return None
    # Remove any unrelated keys from json
    return filter_allowed_keys(schema=json_dict, allowed=allowed_arguments)


# This makes a more descriptive schema but not technically json schema
def tool_to_func_schema(name: str, description: str, params: List[dict]):
    # Construct parameters
    properties = {}
    required = []
    for param in params:
        param_name = param["name"]
        param_type = param["type"]
        param_description = param["description"]
        param_schema = {"type": param_type, "description": param_description}
        # Assign param
        properties[param_name] = param_schema
        required.append(param_name)
    # @TODO Add output type
    func_constrained_schema = {
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
    return json.dumps(func_constrained_schema)


# Conversion func to translate our tool def to a native tool (openai compatible) json schema.
# Used to constrain output to json. Most models will likely use this, but some, like Functionary use other formats.
def tool_to_json_schema(params: List[dict]) -> str:
    # Construct parameters
    properties = {}
    required = []
    for param in params:
        param_name = param["name"]
        param_schema = {"type": param["type"]}
        # Include enum values for Literal types (from options field)
        if param.get("options"):
            param_schema["enum"] = param["options"]
        # Assign param
        properties[param_name] = param_schema
        # Add param to required
        if not param.get("llm_not_required"):
            required.append(param_name)
    object_constrained_schema = {
        "type": "object",
        "properties": properties,
        "required": required,
    }
    return json.dumps(object_constrained_schema)


# Conversion func to translate our tool def to a native tool (typescript) schema
def tool_to_typescript_schema(name: str, description: str, params: List[dict]) -> str:
    # Construct parameters
    properties = ""
    output_type = "any"  # @TODO Define output type
    for param in params:
        # We ignore specific params if its always provided
        if param.get("llm_not_required"):
            continue
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
        descr = schema[pname].get("description", None) or ""
        data_type = schema[pname].get("type", None) or ""
        allowed_values = schema[pname].get("allowed_values", None)
        if index > 0:
            result += "\n\n"
        result += f"### {pname}\n\nDescription: {descr}\nData type: {data_type}"
        if allowed_values:
            result += f"\nAllowed values: {json.dumps(allowed_values)}"
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


def get_provided_args(prompt: str, tool_params: List[dict]):
    provided_arguments = dict()
    for tool_def in tool_params:
        param_name = tool_def.get("name")
        value = tool_def.get("value", None)
        if param_name == "prompt":
            provided_arguments[param_name] = prompt
            continue
        if value:
            provided_arguments[param_name] = value
    return provided_arguments


# Determine allowed arg names (arguments that llm needs to fill in)
def get_llm_required_args(tool_params: List[ToolFunctionParameter]) -> List[str]:
    result = []
    for param in tool_params:
        pname = param.get("name", None)
        val = param.get("value", None)
        is_required = not param.get("llm_not_required")
        # Check each `value` exists, if not then the llm needs to send it.
        # We ignore specific params if its always provided
        if pname and not val and is_required:
            result.append(pname)
    return result


def find_tool_in_response(response: str, tools: List[str]) -> Optional[str]:
    """Attempt to find any tool names at the end of the given response text."""
    print(f"{common.PRNT_API} Searching in response for tools...{tools}", flush=True)
    for tool_name in tools:
        index = response.rfind(re.escape(tool_name))
        if index != -1:
            print(f"{common.PRNT_API} Found tool:{tool_name}", flush=True)
            return tool_name  # Return first match
    return None  # Explicitly return None if no tool is found


def get_built_in_functions() -> dict:
    """Dynamically import all built-in function modules from tools/built_in_functions."""
    import pkgutil
    import importlib
    from tools import built_in_functions

    functions = {}
    # Iterate through all modules in the built_in_functions package
    for _, module_name, ispkg in pkgutil.iter_modules(built_in_functions.__path__):
        # Skip __init__ and any packages (only import .py files)
        if ispkg or module_name.startswith("_"):
            continue
        try:
            # Import the module
            module = importlib.import_module(f"tools.built_in_functions.{module_name}")
            # Add the module to the functions dict using the module name (assuming .py extension as key)
            functions[f"{module_name}.py"] = module
        except Exception as err:
            print(
                f"{common.PRNT_API} Failed to import {module_name}: {err}", flush=True
            )

    return functions
