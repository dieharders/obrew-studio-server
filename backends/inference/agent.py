import json
from typing import List, Type
from sse_starlette.sse import EventSourceResponse
from fastapi import Request, HTTPException
from inference.helpers import GENERATING_TOKENS
from storage.route import get_all_tool_definitions
from tools.tool import Tool
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

        # Handles requests sequentially and streams responses using SSE
        if self.llm.request_queue.qsize() > 0:
            print(f"{common.PRNT_API} Too many requests, please wait.")
            return HTTPException(
                status_code=429, detail="Too many requests, please wait."
            )
        # Add request to queue
        await self.llm.request_queue.put(request)

        # Return tool assisted response if any tools are assigned
        if self.has_tools:
            assigned_tool: ToolDefinition = None
            all_installed_tool_defs: List[ToolDefinition] = (
                get_all_tool_definitions().get("data")
            )
            # @TODO Use the self.active_role attribute and some new logic to determine which tool to select. Right now we are hard-coding to first one
            # Based on active_role, have LLM choose the appropriate tool based on their descriptions and prompt or explicit instruction within the prompt.
            chosen_tool_name = self.tools[0]
            assigned_tool_defs = [
                item for item in all_installed_tool_defs if item["name"] in self.tools
            ]
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
            tool = Tool()
            tool_call_result = await tool.call(
                llm=self.llm, tool_def=assigned_tool, query=prompt
            )
            if streaming:
                payload = {
                    "event": GENERATING_TOKENS,
                    "data": tool_call_result,
                }

                async def chunk_payload():
                    yield json.dumps(payload)

                generator = chunk_payload()
                return EventSourceResponse(generator)
            return tool_call_result

        # Perform normal un-assisted generation
        if prompt_template:
            # Assign the agent's template to the prompt
            query_prompt = prompt_template.replace(common.QUERY_INPUT, prompt)

        match (response_type):
            # Instruct is for Question/Answer (good for tool use, RAG)
            case CHAT_MODES.INSTRUCT.value:
                response = self.llm.text_completion(
                    prompt=query_prompt,
                    system_message=system_message,
                    stream=streaming,
                )
                # Return streaming response
                if streaming:
                    return EventSourceResponse(response)
                # Return complete response
                content = [item async for item in response]
                return content[0].get("data")
            case CHAT_MODES.CHAT.value:
                response = self.llm.text_chat(
                    prompt=query_prompt,
                    system_message=system_message,
                    stream=streaming,
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

        return response
