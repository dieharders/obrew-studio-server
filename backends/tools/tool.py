import os
import re
import json
import importlib.util
from typing import Any, Awaitable, List, Optional, Type
from pydantic import BaseModel
from core import common
from core.classes import ToolDefinition, ToolFunctionParameter, ToolFunctionSchema
from inference.llama_cpp import LLAMA_CPP
from inference.classes import AgentOutput

# The name to look for when loading python module
TOOL_FUNCTION_NAME = "main"
TOOL_NAME = "tool_name"
example_schema = {TOOL_NAME: "name"}
TOOL_CHOICE_SCHEMA = f"```json\n{json.dumps(example_schema, indent=4)}\n```"


# Handles reading, loading and execution of tool functions and their core deps (if any).
# @TODO May have better luck co-ercing Gemma3 using:
# https://www.reddit.com/r/LocalLLaMA/comments/1jauy8d/giving_native_tool_calling_to_gemma_3_or_really/
# and https://www.philschmid.de/gemma-function-calling
class Tool:
    """
    Handles reading, loading and execution of tool functions and their core deps (if any).

    How it works:
    1. Determine if an LLM is needed for generating input arguments.
    2. If an LLM is required we prompt it with the tool schema and pass the (parsed and cleaned) response into the tool function.
    3. Otherwise, pass the tool's params into the tool function.
    4. Finally, return the function's results along with a text version.
    """

    def __init__(self):
        self.filename: str = None
        self.func_definition: ToolFunctionSchema = None
        self.func: Awaitable[Any] = None

    # Used by Workers
    async def choose_tool_from_query(
        self,
        llm: Type[LLAMA_CPP],
        query_prompt: str,
        assigned_tools: List[ToolDefinition],
    ) -> str:
        """
        1. Prompt the llm to read the original prompt and determine if the user wants to use a tool.
        2. Extract the name of the specified tool as json response: {tool_name: ""}.
        """
        # Make names from definitions
        names = []
        for tool in assigned_tools:
            name = tool.get("name")
            if name:
                names.append(name)
        tool_names = ", ".join(names)
        system_message = f'Determine if the user query contains a request or an expression of need for tool use. Identify only one tool name specified in the user query. Return the name of the specified tool in JSON format specified by "Schema".'
        prompt = f'# Tool names:\n\n{tool_names}\n\n# User query:\n\n"{query_prompt}"\n\n## Answer in JSON:\n\nSchema:{TOOL_CHOICE_SCHEMA}'
        # Ask llm to choose
        response = llm.text_completion(
            prompt=prompt,
            system_message=system_message,
            stream=False,
            override_args={"--temp": 0.2},
        )
        content = [item async for item in response]
        data = content[0].get("data")
        # Parse out the json result using either regex or another llm call
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Tool choice structured output:\n{arguments_response_str}"
        )
        try:
            parsed_llm_response = parse_structured_llm_response(
                arguments_str=arguments_response_str, allowed_arguments=[TOOL_NAME]
            )
            chosen_tool: str = parsed_llm_response.get(TOOL_NAME, "")
            print(f"{common.PRNT_API} Chosen tool:{chosen_tool}")
            return chosen_tool
        except Exception as err:
            # Try to recover by just looking in the response for any mention of an assigned tool
            if assigned_tools:
                chosen_tool = find_tool_in_response(
                    response=arguments_response_str, tools=names
                )
                if chosen_tool:
                    return chosen_tool
            raise err

    # Used by Agents
    async def choose_tool_from_description(
        self,
        llm: Type[LLAMA_CPP],
        query_prompt: str,
        assigned_tools: List[ToolDefinition],
    ) -> str:
        """
        1. Create a markdown style string of all assigned tools' description and title.
        2. Prompt the llm with instruction in sys_msg, and a prompt with a command followed by the tool descriptions and original prompt.
        3. Extract the tool name as json response: {tool_name: ""}.
        """
        tool_names = []
        for tool in assigned_tools:
            name = tool.get("name")
            if name:
                tool_names.append(name)
        tool_descriptions = tool_to_markdown(assigned_tools)
        prompt = f"# Tool descriptions:\n\n{tool_descriptions}\n# User query:\n\n{query_prompt}\n\n# Schema:\n\n{TOOL_CHOICE_SCHEMA}"
        system_message = f'Determine the best tool to choose based on each description and the needs of the user query. Return the name of the chosen tool in JSON format that matches the "Schema".'
        # Ask llm to choose
        response = llm.text_completion(
            prompt=prompt,
            system_message=system_message,
            stream=False,
            override_args={"--temp": 0.2},
        )
        content = [item async for item in response]
        data = content[0].get("data")
        # Parse out the json result using either regex or another llm call
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Tool choice structured output:\n{arguments_response_str}"
        )
        try:
            parsed_llm_response = parse_structured_llm_response(
                arguments_str=arguments_response_str, allowed_arguments=[TOOL_NAME]
            )
            chosen_tool = parsed_llm_response.get(TOOL_NAME, "")
            print(f"{common.PRNT_API} Chosen tool:{chosen_tool}")
            return chosen_tool
        except Exception as err:
            # Try to recover by just looking in the response for any mention of an assigned tool
            if assigned_tools:
                chosen_tool = find_tool_in_response(
                    response=arguments_response_str, tools=tool_names
                )
                if chosen_tool:
                    return chosen_tool
            raise err

    # Read the pydantic model for the tool from a file
    def read_function(self, filename: str) -> ToolFunctionSchema:
        self.filename = filename

        try:
            spec = import_tool_function(filename)
            tool_code = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool_code)
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
            # Record output type
            func = getattr(tool_code, TOOL_FUNCTION_NAME)
            return_type = func.__annotations__.get("return")
            return_type_name = func.__annotations__.get("return").__name__
            if return_type_name == "Union":
                # @TODO Make this into an array of strings
                output_types = [str(return_type)]
            else:
                output_types = [return_type_name]

            # Define tool
            self.func_definition = {
                "description": tool_description,
                # Tool parameters configurable by user
                "params": params,
                # Schema for params to pass for tool use
                "params_schema": tool_schema,
                # All required params with example values
                "params_example": example_tool_schema,
                # The return type of the output
                "output_type": output_types,
            }
            return self.func_definition
        except Exception as err:
            print(f"{common.PRNT_API} Error loading tool function: {err}", flush=True)
            raise err

    # Execute the tool function with the provided arguments (if any)
    # Return results in raw (for output to other funcs) and string (for text response) formats
    async def call(
        self,
        tool_def: ToolDefinition,
        llm: Type[LLAMA_CPP] = None,
        query: str = "",
    ) -> AgentOutput:
        self.func_definition = tool_def
        self.func = load_function(tool_def.get("path", None))

        # If llm is not required, skip to func call
        tool_params = tool_def.get("params", None)
        required_llm_arguments = get_llm_required_args(tool_params)
        if len(required_llm_arguments) > 0:
            system_message = 'You are given a name and description of a function along with the input argument schema it expects. Based on this info and the "QUESTION" you are expected to return a JSON formatted string that looks similar to the "SCHEMA EXAMPLE" but with the values replaced. Ensure the JSON is properly formatted and each value is the correct data type according to the "SCHEMA".'
            prompt_template = "# Tool: {tool_name_str}\n\n## Description:\n\n{tool_description_str}\n\n## QUESTION:\n\n{query_str}\n\n## SCHEMA EXAMPLE:\n\n{tool_example_str}\n\n## SCHEMA:\n\n{tool_arguments_str}"
            TOOL_ARGUMENTS = "{tool_arguments_str}"
            TOOL_EXAMPLE_ARGUMENTS = "{tool_example_str}"
            TOOL_NAME = "{tool_name_str}"
            TOOL_DESCRIPTION = "{tool_description_str}"
            # Return schema for llm to respond with
            tool_name_str = self.func_definition.get("name", "Tool")
            tool_description_str = self.func_definition.get("description", "")
            # Parse these to only include data from the required_llm_arguments list
            params_schema_dict = get_required_schema(
                required=required_llm_arguments,
                schema=self.func_definition.get("params_schema", dict()),
            )
            params_example_dict = get_required_examples(
                required=required_llm_arguments,
                example=self.func_definition.get("params_example", dict()),
            )
            # Convert func arguments to machine readable strings
            tool_example_json = json.dumps(params_example_dict, indent=4)
            tool_example_str = f"```json\n{tool_example_json}\n```"
            tool_args_str = schema_to_markdown(params_schema_dict)
            # Inject template args into prompt template
            tool_prompt = prompt_template.replace(common.QUERY_INPUT, query)
            tool_prompt = tool_prompt.replace(TOOL_ARGUMENTS, tool_args_str)
            tool_prompt = tool_prompt.replace(TOOL_EXAMPLE_ARGUMENTS, tool_example_str)
            tool_prompt = tool_prompt.replace(TOOL_NAME, tool_name_str)
            tool_prompt = tool_prompt.replace(TOOL_DESCRIPTION, tool_description_str)
            # Prompt the LLM for a response using the tool's schema.
            # A lower temperature is best for tool use.
            llm_tool_use_response = llm.text_completion(
                prompt=tool_prompt,
                system_message=system_message,
                stream=False,
                override_args={"--temp": 0.2},  # @TODO Set --n-predict, --ctx-size etc?
            )
            content: List[dict] = [item async for item in llm_tool_use_response]
            data: AgentOutput = content[0].get("data")
            # Parse out the json result using regex
            arguments_response_str = data.get("text")
            print(
                f"{common.PRNT_API} Tool call structured output:\n{arguments_response_str}"
            )
            parsed_llm_response = parse_structured_llm_response(
                arguments_str=arguments_response_str,
                allowed_arguments=required_llm_arguments,
            )
            # Call the function with the arguments provided from the llm response, Return results
            print(
                f"{common.PRNT_API} Calling tool function with arguments:\n{json.dumps(parsed_llm_response, indent=4)}"
            )
            func_call_result = await self.func(**parsed_llm_response)
            return dict(raw=func_call_result, text=str(func_call_result))
        # Call function with arguments provided by the tool and/or prompt
        func_arguments = get_provided_args(prompt=query, tool_params=tool_params)
        func_call_result = await self.func(**func_arguments)
        # Return results
        return dict(raw=func_call_result, text=str(func_call_result))


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


# Create a machine readable description of a tool definition.
# Good for explaining each tool to llm so it can make a selection.
def tool_to_markdown(tool_list: List[ToolDefinition], include_code=False):
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
            elif key != "path" and key != "id":
                markdown_string += f"### {key}\n\n{value}\n\n"
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


def get_provided_args(prompt: str, tool_params: dict) -> dict:
    provided_arguments = dict()
    for pname in tool_params:
        value = tool_params[pname].get("value", None)
        if pname == "prompt":
            provided_arguments[pname] = prompt
            continue
        if value:
            provided_arguments[pname] = value


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
    print(f"{common.PRNT_API} Searching in response for tools...{tools}")
    for tool_name in tools:
        match = re.search(re.escape(tool_name), response, re.DOTALL)
        if match:
            print(f"{common.PRNT_API} Found tool:{tool_name}")
            return tool_name  # Return first match immediately
    return None  # Explicitly return None if no tool is found
