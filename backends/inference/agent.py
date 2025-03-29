import json
from typing import List, Type
from sse_starlette.sse import EventSourceResponse
from fastapi import Request
from storage.route import get_all_tool_definitions
from tools.tool import Tool
from inference.helpers import GENERATING_TOKENS, KEY_PROMPT_MESSAGE
from inference.classes import ACTIVE_ROLES, CHAT_MODES, DEFAULT_ACTIVE_ROLE, AgentOutput
from inference.llama_cpp import LLAMA_CPP
from core import common
from core.classes import ToolDefinition


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
    ) -> AgentOutput:
        query_prompt = prompt
        tool_call_result = None

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
            # If llm has native tool calling, allow it to choose from all assigned tools
            if self.llm.is_tool_capable:
                tool_call_result = await tool.native_call(
                    llm=self.llm, tool_defs=assigned_tool_defs, query=prompt
                )
                print(
                    f"{common.PRNT_API} Tool call result:\n{json.dumps(tool_call_result, indent=4)}"
                )
            else:
                # Choose a tool to use
                if len(self.tools) == 1:
                    # Always use the first tool if only one is assigned
                    chosen_tool_name = self.tools[0]
                else:
                    # Based on active_role, have LLM choose the appropriate tool based on their descriptions and prompt or explicit instruction within the prompt.
                    match (self.active_role):
                        case ACTIVE_ROLES.AGENT.value:
                            # Choose the best tool based on each description and the needs of the prompt
                            chosen_tool_name = await tool.choose_tool_from_description(
                                llm=self.llm,
                                query_prompt=query_prompt,
                                assigned_tools=assigned_tool_defs,
                            )
                        case ACTIVE_ROLES.WORKER.value:
                            # @TODO Pass the desired tool as override "selectedTool" with request instead of querying llm in a prompt.
                            # Use only the tool specified in a users query
                            chosen_tool_name = await tool.choose_tool_from_query(
                                llm=self.llm,
                                query_prompt=query_prompt,
                                assigned_tools=assigned_tool_defs,
                            )
                        case _:
                            # Default - None or unknown role specified
                            chosen_tool_name = self.tools[0]
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
            # Return streamed result
            if streaming:
                payload = {
                    "event": GENERATING_TOKENS,
                    "data": tool_call_result,
                }

                async def chunk_payload():
                    yield json.dumps(payload)

                generator = chunk_payload()
                return EventSourceResponse(generator)
            # Return entire result
            return tool_call_result

        ###########################################
        # Or, Perform normal un-assisted generation
        ###########################################
        if prompt_template:
            # Apply the agent's template to the prompt
            query_prompt = prompt_template.replace(KEY_PROMPT_MESSAGE, prompt)

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
                return content[0].get("data")
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
                return content[0].get("data")
            case CHAT_MODES.COLLAB.value:
                # @TODO Add a mode for collaborate
                # ...
                raise Exception("Mode 'collab' is not implemented.")
            case None:
                raise Exception("Check 'mode' is provided.")
            case _:
                raise Exception("No 'mode' or 'collection_names' provided.")
