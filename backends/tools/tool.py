import json
import importlib.util
from fastapi import Request
from typing import List, Optional, Type
from pydantic import BaseModel
from tools.classes import TOOL_SCHEMA_TYPES
from core import common
from core.classes import FastAPIApp, ToolDefinition, ToolFunctionSchema
from inference.llama_cpp import LLAMA_CPP
from inference.classes import AgentOutput, LoadTextInferenceCall, LoadTextInferenceInit
from tools.helpers import (
    KEY_TOOL_NAME,
    KEY_TOOL_PARAMS,
    TOOL_CHOICE_JSON_SCHEMA,
    TOOL_CHOICE_SCHEMA_STR,
    TOOL_FUNCTION_NAME,
    TOOL_OUTPUT_SCHEMA_STR,
    filter_allowed_keys,
    find_tool_in_response,
    get_llm_required_args,
    get_provided_args,
    get_required_schema,
    import_tool_function,
    load_function,
    parse_json_block,
    parse_tool_response,
    schema_to_markdown,
    tool_to_json_schema,
    tool_to_markdown,
    tool_to_typescript_schema,
)
from inference.helpers import (
    KEY_PROMPT_MESSAGE,
    read_event_data,
)


# Handles reading, loading and execution of tool functions and their core deps (if any).
# List of tool models -> https://ollama.com/search?c=tools
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

    def __init__(self, app: FastAPIApp = None, request: Optional[Request] = None):
        self.app = app
        self.request = request
        self.filename: str = None

    # Read the pydantic model for the tool from a file
    def read_function(self, filename: str, tool_name: str) -> ToolFunctionSchema:
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
            properties: dict = schema.get("properties", dict())
            # Make params
            func_name = filename.split(".")[0]
            tool_schema_name = tool_name or func_name
            params = []
            tool_schema = dict()
            for key, value in list(properties.items()):  # Convert items to list
                param = dict(**value)
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
            func_definition = {
                "description": tool_description,
                # Tool parameters configurable by user
                "params": params,
                # Schema for params to pass for tool use
                "params_schema": tool_schema,
                # All required params with example values
                "params_example": example_tool_schema,
                # Tool schemas
                "typescript_schema": tool_to_typescript_schema(
                    name=tool_schema_name,
                    description=tool_description,
                    params=params,
                ),
                "json_schema": tool_to_json_schema(params=params),
                # The return type of the output
                "output_type": output_types,
            }
            return func_definition
        except Exception as err:
            print(f"{common.PRNT_API} Error loading tool function: {err}", flush=True)
            raise err

    # Used by Workers
    async def choose_tool_from_query(
        self,
        llm: Type[LLAMA_CPP],
        query_prompt: str,
        assigned_tools: List[ToolDefinition],
    ):
        """
        1. Prompt the llm to read the original prompt and determine if the user wants to use a tool.
        2. Extract the name of the specified tool from json response.
        """
        # Make names from definitions
        names = []
        for tool in assigned_tools:
            name = tool.get("name")
            if name:
                names.append(name)
        tool_names = ", ".join(names)
        system_message = f'Determine if the user query contains a request or an expression of need for tool use. Identify only one tool name specified in the user query. Return the name of the specified tool in JSON format specified by "Schema".'
        prompt = f'# Tool names:\n\n{tool_names}\n\n# User query:\n\n"{query_prompt}"\n\n## Answer in JSON:\n\nSchema:{TOOL_CHOICE_SCHEMA_STR}'
        # Ask llm to choose
        response = await llm.text_completion(
            prompt=prompt,
            system_message=system_message,
            stream=False,
            request=self.request,
            constrain_json_output=TOOL_CHOICE_JSON_SCHEMA,
        )
        content = [item async for item in response]
        data = read_event_data(content)
        # Parse out the json result using either regex or another llm call
        if not data:
            print(f"{common.PRNT_API} No data returned from LLM response", flush=True)
            return None
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Tool choice structured output:\n{arguments_response_str}",
            flush=True,
        )
        try:
            parsed_llm_response = parse_tool_response(
                json_str=arguments_response_str, allowed_arguments=[KEY_TOOL_NAME]
            )
            # Handle no json block with re-prompt
            if not parsed_llm_response:
                raise Exception("No json found.")
            chosen_tool: str = parsed_llm_response.get(KEY_TOOL_NAME, "")
            print(f"{common.PRNT_API} Chosen tool: {chosen_tool}", flush=True)
            return chosen_tool
        except Exception as err:
            # Try to recover by just looking in the response for any mention of an assigned tool
            if assigned_tools:
                chosen_tool = find_tool_in_response(
                    response=arguments_response_str, tools=names
                )
                if chosen_tool:
                    return chosen_tool
            return None

    # Used by Workers
    async def choose_tool_from_description(
        self,
        llm: Type[LLAMA_CPP],
        query_prompt: str,
        assigned_tools: List[ToolDefinition],
    ) -> str:
        """
        1. Create a markdown style string of all assigned tools' description and title.
        2. Prompt the llm with instruction in sys_msg, and a prompt with a command followed by the tool descriptions and original prompt.
        3. Extract the tool name from json response.
        """
        tool_names = []
        for tool in assigned_tools:
            name = tool.get("name")
            if name:
                tool_names.append(name)
        tool_descriptions = tool_to_markdown(assigned_tools)
        prompt = f"# Tool descriptions:\n\n{tool_descriptions}\n# User query:\n\n{query_prompt}\n\n# Schema:\n\n{TOOL_CHOICE_SCHEMA_STR}"
        system_message = f'Determine the best tool to choose based on each description and the needs of the user query. Return the name of the chosen tool in JSON format that matches the "Schema".'
        # Ask llm to choose
        response = await llm.text_completion(
            prompt=prompt,
            system_message=system_message,
            stream=False,
            request=self.request,
            constrain_json_output=TOOL_CHOICE_JSON_SCHEMA,
        )
        content = [item async for item in response]
        data = read_event_data(content)
        # Parse out the json result using either regex or another llm call
        if not data:
            print(f"{common.PRNT_API} No data returned from LLM response", flush=True)
            return None
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Tool choice structured output:\n{arguments_response_str}"
        )
        try:
            parsed_llm_response = parse_tool_response(
                json_str=arguments_response_str, allowed_arguments=[KEY_TOOL_NAME]
            )
            # Handle no json block with re-prompt
            if not parsed_llm_response:
                raise Exception("No json found.")
            chosen_tool: str = parsed_llm_response.get(KEY_TOOL_NAME, "")
            print(f"{common.PRNT_API} Chosen tool: {chosen_tool}", flush=True)
            return chosen_tool
        except Exception as err:
            # Try to recover by just looking in the response for any mention of an assigned tool
            if assigned_tools:
                chosen_tool = find_tool_in_response(
                    response=arguments_response_str, tools=tool_names
                )
                if chosen_tool:
                    return chosen_tool
            return None

    # @TODO Work-in-progress.
    async def native_call(
        self,
        tool_defs: List[ToolDefinition],
        llm: Type[LLAMA_CPP] = None,
        query: str = "",
        prompt_template: str = None,
        system_message: str = None,
        collections: List[str] = [],
    ):
        tool_schemas = ""
        tool_funcs = dict()
        native_tool_defs = ""
        # Use json schema format for structured output
        for tool_def in tool_defs:
            name = tool_def.get("name")
            tool_funcs[name] = tool_def
            # Determine schema format to use for native func calling
            if llm.tool_schema_type == TOOL_SCHEMA_TYPES.TYPESCRIPT.value:
                def_str = tool_def.get("typescript_schema", "")
            else:
                def_str = tool_def.get("json_schema", "")
            native_tool_defs += f"\n\n{def_str}"
            def_json_str = tool_def.get("json_schema", "")
            tool_schemas += f"\n\n{def_json_str}"
        prompt = f"# Tool descriptions:{tool_schemas}\n# User query:\n\n{query}\n\n# Chosen tool schema:\n\n```json\n"
        tool_system_message = f"Determine if the user query contains a request or an expression of need for a tool and use the chosen tool schema to output in JSON format: {TOOL_OUTPUT_SCHEMA_STR}."
        # Prompt the LLM for a response using the tool's schema.
        # A lower temperature is better for tool use.
        llm_tool_use_response = await llm.text_completion(
            system_message=tool_system_message,
            prompt=prompt,
            stream=False,
            request=self.request,
            native_tool_defs=native_tool_defs,
            # Cant constrain exactly since we dont yet know the chosen tool. Verify this is enough.
            # constrain_json_output=TOOL_OUTPUT_JSON_SCHEMA,
        )
        content: List[dict] = [item async for item in llm_tool_use_response]
        data = read_event_data(content)
        # Parse the output
        if not data:
            print(f"{common.PRNT_API} No data returned from LLM response", flush=True)
            return None
        arguments_response_str = data.get("text")
        print(
            f"{common.PRNT_API} Native tool call structured output:\n{arguments_response_str}",
            flush=True,
        )
        parsed_llm_response = parse_json_block(arguments_response_str)
        # Handle no json
        if not parsed_llm_response:
            return None
        # Get chosen tool from response
        chosen_tool_name = parsed_llm_response.get(KEY_TOOL_NAME)
        tool_choice_def = tool_funcs[chosen_tool_name]
        tool_params = tool_choice_def["params"]
        tool_func = tool_choice_def["path"]
        # Cleanup dict
        allowed_args = get_llm_required_args(tool_params)
        chosen_tool_params: dict = parsed_llm_response.get(KEY_TOOL_PARAMS)
        filter_allowed_keys(schema=chosen_tool_params, allowed=allowed_args)
        # Call function with arguments provided by the tool first
        func_results = await _call_func_with_tool_params(
            app=self.app,
            request=self.request,
            tool_def=tool_choice_def,
            prompt=query,
            prompt_template=prompt_template,
            system_message=system_message,
            model_init_kwargs=llm.model_init_kwargs,
            generate_kwargs=llm.generate_kwargs,
            collections=collections,
        )
        if func_results:
            return func_results
        # Otherwise, Call function with arguments from llm
        else:
            func_call_result = await tool_func(
                **chosen_tool_params, app=self.app, request=self.request
            )
            return dict(raw=func_call_result, text=str(func_call_result))

    # Execute the tool function with the provided arguments (if any)
    # Tool defs are passed to llm are formatted as markdown text.
    # Return results in raw (for output to other funcs) and string (for text response) formats
    async def universal_call(
        self,
        tool_def: ToolDefinition,
        llm: Type[LLAMA_CPP] = None,
        query: str = "",
        prompt_template: str = None,
        system_message: str = None,
        collections: List[str] = [],
    ) -> AgentOutput | None:
        if not tool_def:
            print(
                f"{common.PRNT_API} No tool definition provided to universal_call",
                flush=True,
            )
            return None
        func_results = await _call_func_with_tool_params(
            app=self.app,
            request=self.request,
            tool_def=tool_def,
            prompt=query,
            prompt_template=prompt_template,
            system_message=system_message,
            collections=collections,
            model_init_kwargs=llm.model_init_kwargs,
            generate_kwargs=llm.generate_kwargs,
        )
        # Call function with arguments provided by the tool or llm
        if func_results:
            return func_results
        else:
            TOOL_ARGUMENTS = "{{tool_arguments_str}}"
            # TOOL_EXAMPLE_ARGUMENTS = "{{tool_example_str}}"
            TOOL_NAME_STR = "{{tool_name_str}}"
            TOOL_DESCRIPTION = "{{tool_description_str}}"
            OUTPUT_SCHEMA = "{{output_schema}}"
            tool_params = tool_def.get("params", None)
            required_llm_arguments = get_llm_required_args(tool_params)
            tool_instruction = f'# Tool\n\nYou are given a tool called "{TOOL_NAME_STR}" which does the following:\n{TOOL_DESCRIPTION}\n\n## Parameters\n\nA description of each parameter required by the tool.\n\n{TOOL_ARGUMENTS}\n\n## Instruction\n\nBased on this info and the user query, you are expected to return a JSON schema: {OUTPUT_SCHEMA}. Ensure the JSON is properly formatted and each parameter is the correct data type.'
            tool_prompt = f"QUESTION:\n{KEY_PROMPT_MESSAGE}\n\nANSWER:\n"
            # Return schema for llm to respond with
            tool_name_str = tool_def.get("name", "Tool")
            tool_description_str = tool_def.get("description", "")
            tool_json_schema_str = tool_def.get("json_schema", "")
            # Parse these to only include data from the required_llm_arguments list
            params_schema_dict = get_required_schema(
                required=required_llm_arguments,
                schema=tool_def.get("params_schema", dict()),
            )

            # params_example_dict = get_required_examples(
            #     required=required_llm_arguments,
            #     example=tool_def.get("params_example", dict()),
            # )

            # Convert func arguments to machine readable strings
            # tool_example_json = json.dumps(params_example_dict)
            # tool_example_str = f"```json\n{tool_example_json}\n```"
            tool_args_str = schema_to_markdown(params_schema_dict)
            # Inject template args into system message
            tool_prompt = tool_prompt.replace(KEY_PROMPT_MESSAGE, query or "")
            # tool_prompt = tool_prompt.replace(OUTPUT_SCHEMA, tool_json_schema_str)
            tool_system_message = tool_instruction.replace(
                TOOL_ARGUMENTS, tool_args_str or ""
            )
            # @TODO Do we need an example?
            # tool_system_message = tool_system_message.replace(
            #     TOOL_EXAMPLE_ARGUMENTS, tool_example_str
            # )
            tool_system_message = tool_system_message.replace(
                OUTPUT_SCHEMA, tool_json_schema_str or ""
            )
            tool_system_message = tool_system_message.replace(
                TOOL_NAME_STR, tool_name_str or ""
            )
            tool_system_message = tool_system_message.replace(
                TOOL_DESCRIPTION, tool_description_str or ""
            )
            # Prompt the LLM for a response using the tool's schema.
            # A lower temperature is better for tool use.
            llm_tool_use_response = await llm.text_completion(
                prompt=tool_prompt,
                system_message=tool_system_message,
                stream=False,
                request=self.request,
                constrain_json_output=json.loads(tool_json_schema_str),
            )
            content: List[dict] = [item async for item in llm_tool_use_response]
            data = read_event_data(content)
            # Parse out the json result using regex
            if not data:
                print(
                    f"{common.PRNT_API} No data returned from LLM response", flush=True
                )
                return None
            arguments_response_str = data.get("text")
            print(
                f"{common.PRNT_API} Universal tool call structured output:\n{arguments_response_str}",
                flush=True,
            )
            parsed_llm_response = parse_tool_response(
                json_str=arguments_response_str,
                allowed_arguments=required_llm_arguments,
            )
            # Handle no json block with re-prompt at Agent level (but send back prev response for context)
            if not parsed_llm_response and arguments_response_str:
                # @TODO May want to trim the response so not to blow out context window
                return dict(text=arguments_response_str)
            # Handle no json block with re-prompt at Agent level
            if not arguments_response_str:
                return None
            # Call the function with the arguments provided from the llm response, Return results
            print(
                f"{common.PRNT_API} Calling tool function with arguments:\n{json.dumps(parsed_llm_response, indent=4)}",
                flush=True,
            )
            tool_func = load_function(tool_def.get("path", None))
            func_call_result = await tool_func(
                **parsed_llm_response,
                app=self.app,
                request=self.request,
            )
            return dict(
                raw=func_call_result, text=json.dumps(func_call_result, default=str)
            )


async def _call_func_with_tool_params(
    app: FastAPIApp,
    request: Request,
    tool_def: ToolDefinition,
    prompt: str,
    prompt_template: str,
    system_message: str,
    model_init_kwargs: LoadTextInferenceInit,
    generate_kwargs: LoadTextInferenceCall,
    collections: List[str] = [],
):
    """If tool requires llm params, then return nothing, otherwise return func result."""
    if not tool_def:
        print(
            f"{common.PRNT_API} No tool definition provided to _call_func_with_tool_params",
            flush=True,
        )
        return None
    tool_params = tool_def.get("params", None)
    required_llm_arguments = get_llm_required_args(tool_params)
    # @TODO Allow mixing required llm/tool params
    # If llm params are required, you need the llm to give a struct response...
    if len(required_llm_arguments) > 0:
        return None
    # If llm params not required, call the function directly
    else:
        tool_func = load_function(tool_def.get("path", None))
        if not tool_func:
            raise Exception("No function found.")
        # Call function with arguments provided by the tool and/or prompt
        func_arguments = get_provided_args(prompt=prompt, tool_params=tool_params)
        func_call_result = await tool_func(
            **func_arguments,
            app=app,
            request=request,
            model_init_kwargs=model_init_kwargs,
            generate_kwargs=generate_kwargs,
            prompt_template=prompt_template,
            system_message=system_message,
            memories=collections,
        )
        # Return results
        return dict(
            raw=func_call_result, text=json.dumps(func_call_result, default=str)
        )
