import json
from typing import List, Type
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
from storage.route import get_all_tool_definitions
from tools.tool import Tool
from inference.helpers import KEY_PROMPT_MESSAGE, read_event_data, tool_payload
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
        func_calling: str = None,
    ) -> AgentOutput | SSEResponse:
        tool_call_result = None
        tool_response_prompt = ""
        curr_func_calling = func_calling or self.func_calling  # override allowed

        #########################################################
        # Return tool assisted response if any tools are assigned
        #########################################################
        if self.has_tools:
            tool = Tool(app=self.app, request=request)
            assigned_tool: ToolDefinition = None
            all_installed_tool_defs: List[ToolDefinition] = (
                get_all_tool_definitions().get("data")
            )
            assigned_tool_defs = [
                item for item in all_installed_tool_defs if item["name"] in self.tools
            ]
            # Use native tool calling, choose tool from list of schemas and output args in one-shot
            if curr_func_calling == TOOL_USE_MODES.NATIVE.value:
                tool_call_result = await tool.native_call(
                    llm=self.llm,
                    tool_defs=assigned_tool_defs,
                    query=prompt,
                    prompt_template=prompt_template,
                    system_message=system_message,
                )
            # Choose a tool to use, then execute it
            else:
                # Always use the first tool if only one is assigned
                if len(self.tools) == 1:
                    chosen_tool_name = self.tools[0]
                # Use Universal tool calling
                else:
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
                # Execute the tool. For now tool use is limited to one chosen tool.
                # @TODO In future we could have MultiTool(tools=tools) which can execute multiple chained tools.
                tool_call_result = await tool.universal_call(
                    llm=self.llm,
                    tool_def=assigned_tool,
                    query=prompt,
                    prompt_template=prompt_template,
                    system_message=system_message,
                )
            print(
                f"{common.PRNT_API} Tool call result:\n{json.dumps(tool_call_result, indent=4)}"
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
                # Answer with raw value only
                else:
                    if streaming:

                        async def payload_generator():
                            payload = tool_payload(tool_call_result)
                            yield json.dumps(payload)

                        return EventSourceResponse(payload_generator())
                    return tool_call_result
            else:
                tool_response_prompt = failed_tool_response

        ###########################################
        # Or, Perform normal un-assisted generation
        ###########################################
        p = tool_response_prompt if tool_response_prompt else prompt
        query_prompt = p
        if prompt_template:
            # If a tool call failed handle its prompt, otherwise call normally
            # Apply the agent's template to the prompt
            query_prompt = prompt_template.replace(KEY_PROMPT_MESSAGE, p)

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
