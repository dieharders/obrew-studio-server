import json
from typing import List, Type
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
from storage.route import get_all_tool_definitions
from tools.tool import Tool
from inference.helpers import apply_query_template, read_event_data, tool_payload
from inference.classes import (
    CHAT_MODES,
    TOOL_RESPONSE_MODES,
    DEFAULT_TOOL_USE_MODE,
    TOOL_USE_MODES,
    AgentOutput,
    SSEResponse,
)
from inference.llama_cpp import LLAMA_CPP
from core import common
from core.classes import ToolDefinition


# @TODO Assign "logging": [], "metrics": {} to all final responses.
class Agent:
    def __init__(
        self,
        app,
        llm: Type[LLAMA_CPP],
        tools: List[str],
        func_calling: TOOL_USE_MODES = None,
    ):
        self.app = app
        self.llm = llm
        self.tools = tools
        self.func_calling = func_calling or DEFAULT_TOOL_USE_MODE
        self.has_tools = tools and len(tools) > 0

    # Perform the LLM's operation
    async def call(
        self,
        request: Request,
        prompt: str,
        prompt_template: str,
        streaming: bool,
        system_message: str,
        response_type: str,
        tool_response_type: str,
        collections: List[str] = None,
        func_calling: str = None,
    ) -> AgentOutput | SSEResponse:
        tool_call_result = None
        tool_response_prompt = ""
        curr_func_calling = func_calling or self.func_calling  # override allowed

        #########################################################
        # Return tool assisted response if any tools are assigned
        #########################################################
        text_tool_result = ""
        if self.has_tools:
            tool = Tool(app=self.app, request=request)
            assigned_tool: ToolDefinition = None
            all_installed_tool_defs: List[ToolDefinition] = (
                get_all_tool_definitions().get("data") or []
            )
            assigned_tool_defs = [
                item for item in all_installed_tool_defs if item["name"] in self.tools
            ]

            # Fallback: Load unregistered tools directly from built_in_functions
            found_names = {t["name"] for t in assigned_tool_defs}
            failed_tools = []
            for tool_name in self.tools:
                if tool_name not in found_names:
                    try:
                        schema = tool.read_function(f"{tool_name}.py", tool_name)
                        if schema:
                            assigned_tool_defs.append(
                                {"name": tool_name, "path": f"{tool_name}.py", **schema}
                            )
                            print(
                                f"{common.PRNT_API} Loaded built-in tool: {tool_name}",
                                flush=True,
                            )
                        else:
                            # Schema was None - tool module exists but has no valid schema
                            failed_tools.append(tool_name)
                            print(
                                f"{common.PRNT_API} Warning: Built-in tool '{tool_name}' returned no schema - check tool has valid Params class",
                                flush=True,
                            )
                    except FileNotFoundError:
                        failed_tools.append(tool_name)
                        print(
                            f"{common.PRNT_API} Error: Built-in tool '{tool_name}' not found - verify tool file exists at tools/built_in_functions/{tool_name}.py",
                            flush=True,
                        )
                    except ImportError as e:
                        failed_tools.append(tool_name)
                        print(
                            f"{common.PRNT_API} Error: Failed to import built-in tool '{tool_name}' - {e}",
                            flush=True,
                        )
                    except Exception as e:
                        failed_tools.append(tool_name)
                        print(
                            f"{common.PRNT_API} Error: Unexpected error loading built-in tool '{tool_name}': {type(e).__name__}: {e}",
                            flush=True,
                        )

            # Log summary if any tools failed to load
            if failed_tools:
                print(
                    f"{common.PRNT_API} Warning: {len(failed_tools)} tool(s) failed to load: {failed_tools}. "
                    f"Agent will proceed with {len(assigned_tool_defs)} available tool(s).",
                    flush=True,
                )

            # Use native tool calling, choose tool from list of schemas and output args in one-shot
            if curr_func_calling == TOOL_USE_MODES.NATIVE.value:
                tool_call_result = await tool.native_call(
                    llm=self.llm,
                    tool_defs=assigned_tool_defs,
                    query=prompt,
                    prompt_template=prompt_template,
                    system_message=system_message,
                    collections=collections,
                )
            # Choose a tool to use, then execute it
            else:
                # Always use the first tool if only one is assigned
                if len(assigned_tool_defs) == 1:
                    assigned_tool = assigned_tool_defs[0]
                    print(
                        f"{common.PRNT_API} Single tool assigned, using: {assigned_tool.get('name')}",
                        flush=True,
                    )
                # Use Universal tool calling to choose from multiple tools
                elif len(assigned_tool_defs) > 1:
                    # Choose a tool explicitly or implicitly specified in the user query
                    chosen_tool_name = await tool.choose_tool_from_description(
                        llm=self.llm,
                        query_prompt=prompt,
                        assigned_tools=assigned_tool_defs,
                    )
                    # Get the function associated with the chosen tool name
                    assigned_tool = next(
                        (
                            item
                            for item in assigned_tool_defs
                            if item["name"] == chosen_tool_name
                        ),
                        None,
                    )
                    if not assigned_tool:
                        print(
                            f"{common.PRNT_API} Warning: No matching tool found for chosen tool name: {chosen_tool_name}",
                            flush=True,
                        )
                else:
                    # No valid tool definitions found
                    assigned_tool = None
                    print(
                        f"{common.PRNT_API} Error: No valid tool definitions found for tools: {self.tools}",
                        flush=True,
                    )

                # Execute the tool if one was successfully assigned
                if assigned_tool:
                    # Execute the tool. For now tool use is limited to one chosen tool.
                    # @TODO In future we could have MultiTool(tools=tools) which can execute multiple chained tools.
                    tool_call_result = await tool.universal_call(
                        llm=self.llm,
                        tool_def=assigned_tool,
                        query=prompt,
                        prompt_template=prompt_template,
                        system_message=system_message,
                        collections=collections,
                    )
                else:
                    # Set tool_call_result to None to trigger error handling below
                    tool_call_result = None
            print(
                f"{common.PRNT_API} Tool call result:\n{json.dumps(tool_call_result, indent=4) if tool_call_result else 'None'}"
            )
            # Handle tool response
            failed_tool_response = f"\nFailed to use tool, no JSON block found. The original query:\n{prompt}"
            if tool_call_result:
                raw_tool_result = tool_call_result.get("raw")
                text_tool_result = tool_call_result.get("text")
                # If we get nothing from tool but have a response, answer back with failed response as context.
                if text_tool_result and not raw_tool_result:
                    tool_response_prompt = f"\nFailed to use tool, no JSON block found. Original query:\n{prompt}\n\nThe last response comes from this context:\n{text_tool_result}"
                # If no response from tool and llm, answer back with original prompt.
                elif not text_tool_result and not raw_tool_result:
                    tool_response_prompt = failed_tool_response
                # Answer back with original prompt and tool result
                elif tool_response_type == TOOL_RESPONSE_MODES.ANSWER.value:
                    # Pass thru to "normal" generation logic below
                    tool_response_prompt = f"\n{prompt}\n\nAnswer: {raw_tool_result}"
                # Answer back with raw value only
                else:
                    if streaming:

                        async def payload_generator():
                            payload = tool_payload(tool_call_result)
                            yield json.dumps(payload)

                        return EventSourceResponse(payload_generator())
                    return tool_call_result
            else:
                # Pass thru to "normal" generation logic below
                tool_response_prompt = failed_tool_response

        ###########################################
        # Or, Perform normal un-assisted generation
        ###########################################
        query_prompt = tool_response_prompt if tool_response_prompt else prompt
        if prompt_template:
            # If a tool call failed handle its prompt, otherwise call normally
            # Apply the agent's template to the prompt
            query_prompt = apply_query_template(
                template=prompt_template,
                query=query_prompt,
                existing_answer=text_tool_result,
            )

        match (response_type):
            # Instruct is for Question/Answer (good for tool use, RAG)
            case CHAT_MODES.INSTRUCT.value:
                response = await self.llm.text_completion(
                    prompt=query_prompt,
                    system_message=system_message,
                    stream=streaming,
                    request=request,
                )
                # Return streaming response
                if streaming:
                    return EventSourceResponse(response)
                # Return complete response
                content = [item async for item in response]
                data = read_event_data(content)
                return data
            # Long running conversation that remembers discussion history
            case CHAT_MODES.CHAT.value:
                response = await self.llm.text_chat(
                    prompt=query_prompt,
                    system_message=system_message,
                    stream=streaming,
                    request=request,
                )
                # Return streaming response
                if streaming:
                    return EventSourceResponse(response)
                # Return complete response
                content = [item async for item in response]
                data = read_event_data(content)
                return data
            case CHAT_MODES.COLLAB.value:
                # @TODO Add a mode for collaborate
                # ...
                raise Exception("Mode 'collab' is not implemented.")
            case None:
                raise Exception("Check 'mode' is provided.")
            case _:
                raise Exception("No 'mode' or 'collection_names' provided.")
