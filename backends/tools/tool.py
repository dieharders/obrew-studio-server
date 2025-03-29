import json
from enum import Enum
import importlib.util
from fastapi import Request
from typing import Any, Awaitable, List, Optional, Type
from pydantic import BaseModel
from tools.helpers import (
    TOOL_FUNCTION_NAME,
    find_tool_in_response,
    get_llm_required_args,
    get_provided_args,
    get_required_examples,
    get_required_schema,
    import_tool_function,
    load_function,
    parse_structured_llm_response,
    schema_to_markdown,
    tool_to_json_schema,
    tool_to_markdown,
    tool_to_typescript_schema,
)
from core import common
from core.classes import ToolDefinition, ToolFunctionSchema
from inference.helpers import KEY_PROMPT_MESSAGE
from inference.llama_cpp import LLAMA_CPP
from inference.classes import AgentOutput

# The name to look for when loading python module
TOOL_NAME = "tool_name"
example_schema = {TOOL_NAME: "name"}
TOOL_CHOICE_SCHEMA = f"```json\n{json.dumps(example_schema, indent=4)}\n```"


class TOOL_SCHEMA_TYPE(str, Enum):
    TYPESCRIPT = "typescript"
    JSON = "json"  # openai format


# Handles reading, loading and execution of tool functions and their core deps (if any).
# @TODO May have better luck coercing Gemma3 using:
# https://www.reddit.com/r/LocalLLaMA/comments/1jauy8d/giving_native_tool_calling_to_gemma_3_or_really/
# and https://www.philschmid.de/gemma-function-calling
# List of Ollama tool models -> https://ollama.com/search?c=tools
# https://huggingface.co/cfahlgren1/natural-functions -- https://ollama.com/calebfahlgren/natural-functions
class Tool:
    """
    Handles reading, loading and execution of tool functions and their core deps (if any).

    How it works:
    1. Determine if an LLM is needed for generating input arguments.
    2. If an LLM is required we prompt it with the tool schema and pass the (parsed and cleaned) response into the tool function.
    3. Otherwise, pass the tool's params into the tool function.
    4. Finally, return the function's results along with a text version.
    """

    def __init__(self, request: Optional[Request] = None):
        self.request = request
        self.filename: str = None
        # @TODO Dont think we need these here
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
        response = await llm.text_completion(
            prompt=prompt,
            system_message=system_message,
            stream=False,
            request=self.request,
        )
        content = [item async for item in response]
        data = content[0].get("data")
        # Parse out the json result using either regex or another llm call
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Tool choice structured output:\n{arguments_response_str}",
            flush=True,
        )
        try:
            parsed_llm_response = parse_structured_llm_response(
                arguments_str=arguments_response_str, allowed_arguments=[TOOL_NAME]
            )
            chosen_tool: str = parsed_llm_response.get(TOOL_NAME, "")
            print(f"{common.PRNT_API} Chosen tool:{chosen_tool}", flush=True)
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
        response = await llm.text_completion(
            prompt=prompt,
            system_message=system_message,
            stream=False,
            request=self.request,
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
        """Reads a tool's pydantic function from python file, constructs schemas, and outputs a tool schema definition."""
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
            func_name = filename.split(".")[0]
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
                # @TODO Make "return_type" this into an array of strings
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
                # Tool schemas
                "typescript_schema": tool_to_typescript_schema(
                    name=func_name,
                    description=tool_description,
                    params=params,
                ),
                "json_schema": tool_to_json_schema(
                    name=func_name,
                    description=tool_description,
                    params=params,
                ),
                # The return type of the output
                "output_type": output_types,
            }
            return self.func_definition
        except Exception as err:
            print(f"{common.PRNT_API} Error loading tool function: {err}", flush=True)
            raise err

    # @TODO Work-in-progress. Difficulty getting Functionary to output args.
    async def native_call(
        self,
        tool_defs: List[ToolDefinition],
        llm: Type[LLAMA_CPP] = None,
        query: str = "",
    ):
        native_tool_defs = ""
        for tool_def in tool_defs:
            # Determine schema format to use for native func calling
            if llm.tool_schema_type == TOOL_SCHEMA_TYPE.TYPESCRIPT.value:
                def_str = tool_def.get("typescript_schema", "")
            else:
                def_str = tool_def.get("json_schema", "")
            native_tool_defs += f"\n\n{def_str}"
        # Prompt the LLM for a response using the tool's schema.
        # A lower temperature is better for tool use.
        llm_tool_use_response = await llm.text_completion(
            prompt=query,
            system_message="",
            stream=False,
            request=self.request,
            native_tool_defs=native_tool_defs,
        )
        content: List[dict] = [item async for item in llm_tool_use_response]
        # Parse the output
        data: AgentOutput = content[0].get("data")
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Native tool call structured output:\n{arguments_response_str}",
            flush=True,
        )
        parsed_llm_response = json.loads(arguments_response_str)
        func_call_result = await self.func(**parsed_llm_response)
        return dict(raw=func_call_result, text=str(func_call_result))

    # Execute the tool function with the provided arguments (if any)
    # Return results in raw (for output to other funcs) and string (for text response) formats
    async def universal_call(
        self,
        tool_def: ToolDefinition,
        llm: Type[LLAMA_CPP] = None,
        query: str = "",
    ) -> AgentOutput:
        self.func_definition = tool_def
        self.func = load_function(tool_def.get("path", None))
        # Determine if llm is required, if so call the function
        tool_params = tool_def.get("params", None)
        required_llm_arguments = get_llm_required_args(tool_params)
        if len(required_llm_arguments) > 0:
            system_message = 'You are given a name and description of a function along with the input argument schema it expects. Based on this info and the "QUESTION" you are expected to return a JSON formatted string that looks similar to the "SCHEMA EXAMPLE" but with the values replaced. Ensure the JSON is properly formatted and each value is the correct data type according to the "SCHEMA".'
            universal_tool_template = "# Tool: {{tool_name_str}}\n\n## Description:\n\n{{tool_description_str}}\n\n## QUESTION:\n\n{{user_prompt}}\n\n## SCHEMA EXAMPLE:\n\n{{tool_example_str}}\n\n## SCHEMA:\n\n{{tool_arguments_str}}"
            TOOL_ARGUMENTS = "{{tool_arguments_str}}"
            TOOL_EXAMPLE_ARGUMENTS = "{{tool_example_str}}"
            TOOL_NAME = "{{tool_name_str}}"
            TOOL_DESCRIPTION = "{{tool_description_str}}"
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
            # Inject template args into prompt template (Universal func calling)
            tool_prompt = universal_tool_template.replace(KEY_PROMPT_MESSAGE, query)
            tool_prompt = tool_prompt.replace(TOOL_ARGUMENTS, tool_args_str)
            tool_prompt = tool_prompt.replace(TOOL_EXAMPLE_ARGUMENTS, tool_example_str)
            tool_prompt = tool_prompt.replace(TOOL_NAME, tool_name_str)
            tool_prompt = tool_prompt.replace(TOOL_DESCRIPTION, tool_description_str)
            # Prompt the LLM for a response using the tool's schema.
            # A lower temperature is better for tool use.
            llm_tool_use_response = await llm.text_completion(
                prompt=tool_prompt,
                system_message=system_message,
                stream=False,
                request=self.request,
            )
            content: List[dict] = [item async for item in llm_tool_use_response]
            data: AgentOutput = content[0].get("data")
            # Parse out the json result using regex
            arguments_response_str = data.get("text")
            print(
                f"{common.PRNT_API} Universal tool call structured output:\n{arguments_response_str}",
                flush=True,
            )
            parsed_llm_response = parse_structured_llm_response(
                arguments_str=arguments_response_str,
                allowed_arguments=required_llm_arguments,
            )
            # Call the function with the arguments provided from the llm response, Return results
            print(
                f"{common.PRNT_API} Calling tool function with arguments:\n{json.dumps(parsed_llm_response, indent=4)}",
                flush=True,
            )
            func_call_result = await self.func(**parsed_llm_response)
            return dict(raw=func_call_result, text=str(func_call_result))
        else:
            # Call function with arguments provided by the tool and/or prompt
            func_arguments = get_provided_args(prompt=query, tool_params=tool_params)
            func_call_result = await self.func(**func_arguments)
            # Return results
            return dict(raw=func_call_result, text=str(func_call_result))
