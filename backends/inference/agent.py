import json
from typing import List, Type
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
from storage.route import get_all_tool_definitions
from tools.tool import Tool
from inference.helpers import KEY_PROMPT_MESSAGE, read_event_data, tool_payload
from inference.classes import (
    ACTIVE_ROLES,
    CHAT_MODES,
    DEFAULT_ACTIVE_ROLE,
    TOOL_RESPONSE_MODES,
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
        llm: Type[LLAMA_CPP],
        tools: List[str],
        active_role: ACTIVE_ROLES = DEFAULT_ACTIVE_ROLE,
    ):
        self.llm = llm
        self.tools = tools
        self.active_role = active_role
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
    ) -> AgentOutput | SSEResponse:
        query_prompt = prompt
        tool_call_result = None
        tool_response_prompt = ""

        #########################################################
        # Return tool assisted response if any tools are assigned
        #########################################################
        if self.has_tools:
            tool = Tool(request)
            assigned_tool: ToolDefinition = None
            all_installed_tool_defs: List[ToolDefinition] = (
                get_all_tool_definitions().get("data")
            )
            assigned_tool_defs = [
                item for item in all_installed_tool_defs if item["name"] in self.tools
            ]
            # If llm has native tool calling, allow it to choose from all assigned tools. Inject tool choices into prompt.
            if self.llm.is_tool_capable:
                tool_call_result = await tool.native_call(
                    llm=self.llm, tool_defs=assigned_tool_defs, query=prompt
                )
            # Choose tool from list of schemas and output args in one-shot
            elif self.active_role == ACTIVE_ROLES.AGENT.value:
                tool_call_result = await tool.choose_and_call(
                    llm=self.llm, tool_defs=assigned_tool_defs, query=prompt
                )
            # Choose a tool to use
            else:
                # Always use the first tool if only one is assigned
                if len(self.tools) == 1:
                    chosen_tool_name = self.tools[0]
                # Based on active_role, have LLM choose the appropriate tool based on their descriptions and prompt or explicit instruction within the prompt.
                elif self.active_role == ACTIVE_ROLES.WORKER.value:
                    # @TODO Pass the desired tool as override "chosenTool" with request instead of querying llm in a prompt?
                    # Choose a tool explicitly or implicitly specified in the user query
                    chosen_tool_name = await tool.choose_tool_from_query(
                        llm=self.llm,
                        query_prompt=query_prompt,
                        assigned_tools=assigned_tool_defs,
                    )
                    # Choose the best tool based on each description and the needs of the prompt
                    # chosen_tool_name = await tool.choose_tool_from_description(...)
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
                    llm=self.llm, tool_def=assigned_tool, query=prompt
                )
            print(
                f"{common.PRNT_API} Tool call result:\n{json.dumps(tool_call_result, indent=4)}"
            )
            # Handle tool responses
            #
            # If we get nothing from tool, answer back with failed response as context
            if not tool_call_result:
                # Apply the agent's template to the prompt along with original prompt and answer
                if prompt_template:
                    tool_response_prompt = f"\nFailed to use tool, no JSON block found. The last response comes from this context:\n{prompt}"
            # Answer back with original prompt and tool result
            elif tool_response_type == TOOL_RESPONSE_MODES.ANSWER.value:
                # Apply the agent's template to prompt
                if prompt_template:
                    raw_answer = tool_call_result.get("raw")
                    tool_response_prompt = f"\n{prompt}\n\nAnswer: {raw_answer}"
                # Pass thru to "normal" generation logic below
            else:
                # Return raw value only
                if streaming:

                    async def payload_generator():
                        payload = tool_payload(tool_call_result)
                        yield json.dumps(payload)

                    return EventSourceResponse(payload_generator())
                return tool_call_result

        ###########################################
        # Or, Perform normal un-assisted generation
        ###########################################
        if prompt_template:
            p = tool_response_prompt if tool_response_prompt else prompt
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
