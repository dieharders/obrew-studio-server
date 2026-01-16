from typing import Optional, List, Sequence
from typing_extensions import TypedDict
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


def inference_call_to_cli_args(model) -> dict:
    """Convert inference Pydantic model (LoadTextInferenceCall) to llama.cpp CLI args."""
    kwargs = {
        "--mirostat-ent": model.mirostat_tau,
        "--top-k": model.top_k,
        "--top-p": model.top_p,
        "--min-p": model.min_p,
        "--repeat-penalty": model.repeat_penalty,
        "--presence-penalty": model.presence_penalty,
        "--frequency-penalty": model.frequency_penalty,
        "--temp": model.temperature,
        "--n-predict": model.max_tokens,
    }
    if model.stop:
        kwargs["--reverse-prompt"] = model.stop
    if model.grammar:
        kwargs["--grammar"] = model.grammar
    return kwargs


def apply_query_template(
    template: str,
    query: str,
    context_list: List[str] = [],
    new_chunk_text: str = "",
    schema: str = "",
    dialect: str = "",
    existing_answer: str = "",
    instruction_str: str = "",
    max_knowledge_triplets: int = 1,
    branching_factor: int = 1,
    max_keywords: int = 1,
    num_chunks: int = 1,
):
    """Return question text for retrieval (RAG)"""

    result = template.replace(KEY_PROMPT_MESSAGE, query)
    if len(context_list) != 0:
        context_str = "\n".join(context_list)
        result = result.replace(KEY_CONTEXT_MESSAGE, context_str)
    result = result.replace("{{num_chunks}}", str(num_chunks))
    result = result.replace("{{max_keywords}}", str(max_keywords))
    result = result.replace("{{new_chunk_text}}", new_chunk_text or "")
    result = result.replace("{{branching_factor}}", str(branching_factor))
    result = result.replace("{{schema}}", schema or "")
    result = result.replace("{{dialect}}", dialect or "")
    result = result.replace("{{existing_answer}}", existing_answer or "")
    result = result.replace("{{max_knowledge_triplets}}", str(max_knowledge_triplets))
    result = result.replace("{{instruction_str}}", instruction_str or "")
    return result


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
                prompt = prompt.replace(KEY_TOOL_MESSAGE, native_tool_defs or "")
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


# Vision/Multi-modal utilities
import base64
import os

# Security limits for image processing
MAX_BASE64_SIZE_MB = 50  # Maximum base64 string size in MB
MAX_IMAGE_DIMENSION = 8192  # Maximum width/height in pixels (8K resolution)


def decode_base64_image(
    base64_string: str,
    temp_dir: str,
    max_size_mb: float = MAX_BASE64_SIZE_MB,
) -> str:
    """
    Decode a base64 encoded image and save to a temporary file.
    Returns the path to the temporary file.

    Args:
        base64_string: Base64 encoded image data (optionally with data URL prefix)
        temp_dir: Directory to save the temporary file
        max_size_mb: Maximum allowed size of base64 string in MB (default 50MB)

    Returns:
        Path to the temporary file

    Raises:
        ValueError: If the base64 string exceeds size limits
        Exception: If decoding fails
    """
    try:
        # Remove data URL prefix if present (e.g., "data:image/png;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        # Security: Check base64 string size before decoding
        max_size_bytes = int(max_size_mb * 1024 * 1024)
        if len(base64_string) > max_size_bytes:
            raise ValueError(
                f"Base64 image data exceeds maximum size of {max_size_mb}MB "
                f"(got {len(base64_string) / (1024 * 1024):.1f}MB)"
            )

        image_data = base64.b64decode(base64_string)

        # Detect image format from magic bytes
        if image_data[:8] == b"\x89PNG\r\n\x1a\n":
            ext = ".png"
        elif image_data[:2] == b"\xff\xd8":
            ext = ".jpg"
        elif image_data[:4] == b"GIF8":
            ext = ".gif"
        elif image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP":
            ext = ".webp"
        else:
            ext = ".png"  # Default to PNG

        # Generate unique filename
        unique_id = os.urandom(8).hex()
        temp_path = os.path.join(temp_dir, f"vision_input_{unique_id}{ext}")

        # Write image to file
        with open(temp_path, "wb") as f:
            f.write(image_data)

        return temp_path
    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        print(f"{common.PRNT_LLAMA} Error decoding base64 image: {e}")
        raise Exception(f"Failed to decode base64 image: {e}")


def preprocess_image(
    input_path: str,
    temp_dir: str,
    max_resolution: int = 1024,
    max_dimension: int = MAX_IMAGE_DIMENSION,
) -> str:
    """
    Preprocess image for llama.cpp vision compatibility.

    This function addresses several issues:
    1. Re-encodes images to fix stb_image compatibility issues with macOS screenshots
       (stb_image has known issues with PNG files saved on macOS)
    2. Resizes large images to reduce memory usage and improve processing speed
    3. Converts to RGB color space (removes alpha channel)
    4. Supports additional formats like WebP that stb_image doesn't support
    5. Validates image dimensions for security

    Args:
        input_path: Path to the input image file
        temp_dir: Directory to save the preprocessed image
        max_resolution: Maximum width/height for output (maintains aspect ratio)
        max_dimension: Maximum allowed input dimension (security limit)

    Returns:
        Path to the preprocessed image file

    Raises:
        ValueError: If image dimensions exceed security limits
    """
    try:
        from PIL import Image

        with Image.open(input_path) as img:
            # Security: Validate image dimensions before processing
            if img.width > max_dimension or img.height > max_dimension:
                raise ValueError(
                    f"Image dimensions ({img.width}x{img.height}) exceed maximum "
                    f"allowed dimension of {max_dimension}px. Please resize the image."
                )
            # Convert to RGB (handles RGBA, P, L, LA modes)
            if img.mode in ("RGBA", "LA"):
                # Create white background and paste image with alpha
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode == "P":
                # Palette mode - convert to RGBA first to handle transparency
                img = img.convert("RGBA")
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            # Resize if too large (maintains aspect ratio)
            if img.width > max_resolution or img.height > max_resolution:
                img.thumbnail(
                    (max_resolution, max_resolution), Image.Resampling.LANCZOS
                )
                print(
                    f"{common.PRNT_LLAMA} Resized image to {img.width}x{img.height}",
                    flush=True,
                )

            # Save as PNG (lossless, no compression artifacts)
            unique_id = os.urandom(8).hex()
            output_path = os.path.join(temp_dir, f"vision_processed_{unique_id}.png")
            img.save(output_path, "PNG")

            return output_path

    except ValueError:
        # Re-raise security/validation errors - don't fall back
        raise
    except Exception as e:
        print(f"{common.PRNT_LLAMA} Error preprocessing image: {e}", flush=True)
        # Fallback: return original path if preprocessing fails (non-security errors)
        print(
            f"{common.PRNT_LLAMA} Falling back to original image: {input_path}",
            flush=True,
        )
        return input_path


def cleanup_temp_images(image_paths: List[str]):
    """
    Remove temporary image files created during vision inference.
    Only removes files that match the vision_input_ or vision_processed_ pattern.
    """
    for path in image_paths:
        try:
            if os.path.exists(path) and (
                "vision_input_" in path or "vision_processed_" in path
            ):
                os.remove(path)
                print(f"{common.PRNT_LLAMA} Cleaned up temp image: {path}")
        except Exception as e:
            print(f"{common.PRNT_LLAMA} Error cleaning up temp image {path}: {e}")
