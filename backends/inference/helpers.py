from typing import Optional
from typing import List, Optional, Sequence
from inference.classes import DEFAULT_SYSTEM_MESSAGE
from llama_index.core.base.llms.types import ChatMessage, MessageRole


# Convert input to the target prompt format for a given model.
# Note completions dont utilize system message (b/c neither do Instruct type models) so we combine it with input msg if present.
def apply_prompt_template(
    input_message: Optional[str] = "",
    system_message: Optional[str] = None,
    template_str: Optional[str] = None,  # Model specific template
):
    if template_str:
        txt = input_message.strip()
        # @TODO Do this for /completions: template_str.replace("{{system_message}}", txt) instead ?
        # ...maybe make sep func for this

        # @TODO Do this instead for chat since system_message is sent seperatly
        if system_message:
            txt = f"{system_message.strip()}\n\n{input_message.strip()}"
        # Format to specified template
        return template_str.replace("{{prompt}}", txt)
    else:
        # Dont format if no template supplied
        if system_message:
            return f"{system_message.strip()}\n{input_message.strip()}"
        return f"{input_message.strip()}"


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


# Not currently used
#
# https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
def format_chat_to_prompt(
    user_message, chat_history=[], system_message=DEFAULT_SYSTEM_MESSAGE
):
    """Manually formats a chat conversation like `--chat-template`."""

    # @TODO Use if chat_history is provided
    formatted_history = "\n".join(
        f"User: {msg['user']}\nAssistant: {msg['assistant']}" for msg in chat_history
    )

    return f"""
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        """


# Convert structured chat conversation to prompt (str)
# @TODO Could also use: from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt
def messages_to_prompt(
    messages: Sequence[ChatMessage],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
    template: Optional[dict] = {},  # Model specific template
) -> str:
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
