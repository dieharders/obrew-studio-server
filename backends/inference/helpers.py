from typing import Optional
from typing_extensions import TypedDict
from typing import List, Optional, Sequence
from core import common
from inference.classes import (
    DEFAULT_SYSTEM_MESSAGE,
    AgentOutput,
    ChatMessage,
    MessageRole,
    SSEResponse,
)

# Event names
GENERATING_TOKENS = "GENERATING_TOKENS"
GENERATING_CONTENT = "GENERATING_CONTENT"
FEEDING_PROMPT = "FEEDING_PROMPT"

# These are the supported template keys
KEY_SYS_MESSAGE = "{{system_message}}"
KEY_USER_MESSAGE = "{{user_message}}"  # Final message to send with prompt template applied to user's prompt
KEY_PROMPT_MESSAGE = "{{user_prompt}}"  # User's query
KEY_TOOL_MESSAGE = "{{tool_defs}}"
KEY_CONTEXT_MESSAGE = "{{context_str}}"  # used by tools and RAG


class Message_Template(TypedDict):
    system: str
    user: str


def sanitize_kwargs(kwargs: dict) -> list[str]:
    arr = []
    for key, value in kwargs.items():
        if isinstance(value, bool):
            if value == True:
                arr.append(key)
            else:
                pass
        else:
            arr.extend([f"{key}", str(value)])
    return arr


# This assumes one turn convo (question/answer) so system msg is included. Feed result to /completion.
def completion_to_prompt(
    user_message: str,
    system_message: Optional[str] = "",
    messageFormat: Optional[Message_Template] = None,
    native_tool_defs: Optional[str] = "",
):
    """Convert user message to prompt by applying the model's format."""

    try:
        sys_msg = system_message  # or DEFAULT_SYSTEM_MESSAGE
        prompt = ""

        if messageFormat:
            # Check if messageFormat includes a system message token and assign system message
            if messageFormat["system"]:
                if messageFormat["system"].find(KEY_SYS_MESSAGE) != -1 and sys_msg:
                    prompt = messageFormat["system"].replace(
                        KEY_SYS_MESSAGE, sys_msg.strip()
                    )
                else:
                    # A template without token means it has sys message baked in
                    prompt = messageFormat["system"]
            elif sys_msg:
                prompt = sys_msg.strip()
            # Check if messageFormat includes tool_defs
            if prompt.find(KEY_TOOL_MESSAGE) != -1:
                prompt = prompt.replace(KEY_TOOL_MESSAGE, native_tool_defs)
            # Check if messageFormat includes a user message
            if (
                messageFormat["user"]
                and messageFormat["user"].find(KEY_USER_MESSAGE) != -1
            ):
                # Add prompt to sys message
                prompt += messageFormat["user"].replace(
                    KEY_USER_MESSAGE, user_message.strip()
                )
            # @TODO Check if messageFormat includes {{context_str}}
            return prompt
        else:
            # No template, combine sys and user messages
            if sys_msg:
                return f"{sys_msg.strip()}\n\n{user_message.strip()}"
            else:
                return user_message.strip()
    except Exception as e:
        print(f"{common.PRNT_LLAMA} Error: {e}")


# def chat_to_prompt(
#     user_message: str,
#     messageFormat: Optional[Message_Template] = None,
# ):
#     """Convert a single user chat message to prompt by applying the model's format."""
#     command = user_message.strip()
#     if messageFormat:
#         # Check if template includes a user message
#         if messageFormat["user"] and messageFormat["user"].find(KEY_USER_MESSAGE) != -1:
#             command = messageFormat["user"].replace(
#                 KEY_USER_MESSAGE, user_message.strip()
#             )
#         # @TODO Check if messageFormat includes {{tool_defs}}
#         # @TODO Check if messageFormat includes {{context_str}}
#     return f"{command}\\"


# Convert structured chat conversation to prompt (str). Result would be fed to a /completion after loading its kv cache.
# @TODO Yet to be implemented.
def messages_to_prompt(
    messages: Sequence[ChatMessage],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
    template: Optional[dict] = {},  # Model specific template
) -> str:
    """Convert an array message history into a prompt by applying the model's template."""

    # (end tokens, structure, etc)
    # @TODO Pass these in from UI model_configs.json (values found in config.json of HF model card)
    BOS = template["BOS"] or ""  # begin string
    EOS = template["EOS"] or ""  # end of string
    B_INST = template["B_INST"] or ""  # begin instruction (user prompt)
    E_INST = template["E_INST"] or ""  # end instruction (user prompt)
    B_SYS = template["B_SYS"] or ""  # begin system instruction
    E_SYS = template["E_SYS"] or ""  # end system instruction

    string_messages: List[str] = []
    if messages[0].role == MessageRole.SYSTEM.value:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER.value

        if i == 0:
            # make sure system prompt is included at the start
            str_message = f"{BOS} {B_INST} {system_message_str} "
        else:
            # end previous user-assistant interaction
            string_messages[-1] += f" {EOS}"
            # no need to include system prompt
            str_message = f"{BOS} {B_INST} "

        # include user message content
        str_message += f"{user_message.content} {E_INST}"

        if len(messages) > (i + 1):
            # if assistant message exists, add to str_message
            assistant_message = messages[i + 1]
            assert assistant_message.role == MessageRole.ASSISTANT.value
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)


def tool_payload(data: str) -> SSEResponse:
    payload = {
        "event": GENERATING_CONTENT,
        "data": data,
    }
    return payload


def token_payload(text: str) -> SSEResponse:
    chunk = {"text": text}
    payload = {
        "event": GENERATING_TOKENS,
        "data": chunk,
    }
    return payload


# Final text response. This should replace all previous text.
def content_payload(text: str) -> SSEResponse:
    # @TODO Add other attrs here...
    content = {"text": text}
    payload = {
        "event": GENERATING_CONTENT,
        "data": content,
    }
    return payload


def event_payload(event_name: str) -> SSEResponse:
    return {"event": event_name}


# Find the "data" in content matching the event
def read_event_data(data_events: List[dict]) -> AgentOutput:
    dataIndex = 0
    for index, event in enumerate(data_events):
        if event.get("event") == GENERATING_CONTENT and event.get("data"):
            dataIndex = index
            break
    data: dict = data_events[dataIndex].get("data")
    if not data:
        raise Exception("No data returned.")
    return data
