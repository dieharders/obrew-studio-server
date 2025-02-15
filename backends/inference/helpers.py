from typing import Optional
from typing_extensions import TypedDict
from typing import List, Optional, Sequence
from core import common
from inference.classes import DEFAULT_SYSTEM_MESSAGE, ChatMessage, MessageRole

# These are the supported template keys
KEY_SYS_MESSAGE = "{{system_message}}"
KEY_USER_MESSAGE = "{{prompt}}"


class Message_Template(TypedDict):
    system: str
    user: str


# This assumes one turn convo (question/answer) so system msg is included. Feed result to /completion.
def completion_to_prompt(
    user_message: str,
    system_message: Optional[str] = None,
    template: Optional[Message_Template] = None,
):
    """Convert user message to prompt by applying the model's template."""

    try:
        sys_msg = system_message or DEFAULT_SYSTEM_MESSAGE
        prompt = ""

        if template:
            # Check if template includes a system message token and assign system message
            if template["system"] and template["system"].find(KEY_SYS_MESSAGE) != -1:
                prompt = template["system"].replace(KEY_SYS_MESSAGE, sys_msg.strip())
            # Check if template includes a user message
            if template["user"] and template["user"].find(KEY_USER_MESSAGE) != -1:
                prompt += template["user"].replace(
                    KEY_USER_MESSAGE, f"{sys_msg.strip()}\n\n{user_message.strip()}"
                )
            return prompt
        else:
            # No template, combine sys and user messages
            return f"{sys_msg.strip()}\n\n{user_message.strip()}"
    except Exception as e:
        print(f"{common.PRNT_LLAMA} Error: {e}")


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
    if messages[0].role == MessageRole.SYSTEM:
        # pull out the system message (if it exists in messages)
        system_message_str = messages[0].content or ""
        messages = messages[1:]
    else:
        system_message_str = system_prompt

    system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

    for i in range(0, len(messages), 2):
        # first message should always be a user
        user_message = messages[i]
        assert user_message.role == MessageRole.USER

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
            assert assistant_message.role == MessageRole.ASSISTANT
            str_message += f" {assistant_message.content}"

        string_messages.append(str_message)

    return "".join(string_messages)
