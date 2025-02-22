###
# Llama-Index allows us to search embeddings in a db and perform queries on them.
# It wraps llama-cpp-python so we can run inference from here as well.
###
import os
import json
from typing import List, Optional, Sequence
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.callbacks import CallbackManager
from inference.classes import CHAT_MODES
from core import common, classes

# These generic helper funcs wont add End_of_seq tokens etc but construct the Prompt/Message
# from llama_index.llms.generic_utils import messages_to_prompt

CONTEXT_INPUT = "{context_str}"  # used by tools and RAG
QUERY_INPUT = "{query_str}"  # the user's prompt
# More templates found here: https://github.com/run-llama/llama_index/blob/main/llama_index/prompts/default_prompts.py
DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant that answers questions in a friendly manner. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
"""

# Helpers


# Format the prompt for completion
# @TODO Could also use: from llama_index.llms.llama_cpp.llama_utils import completion_to_prompt
def completion_to_prompt(
    completion: Optional[str] = "",
    system_prompt: Optional[str] = None,
    template_str: Optional[str] = None,  # Model specific template
):
    if not system_prompt or len(system_prompt.strip()) == 0:
        system_prompt = DEFAULT_SYSTEM_MESSAGE

    # print(
    #     f"\nprompt:\n{completion}\n\nsystem_prompt:\n{system_prompt}template:\n{template_str}",
    #     flush=True,
    # )

    # Format to default spec if no template supplied
    if not template_str:
        return f"{system_prompt.strip()} {completion.strip()}"

    # Format to specified template
    prompt_str = template_str.replace("{prompt}", completion.strip())
    completion_str = prompt_str.replace("{system_message}", system_prompt.strip())
    return completion_str


# Format the prompt for chat conversations
# @TODO Could also use: from llama_index.llms.llama_cpp.llama_utils import messages_to_prompt
def messages_to_prompt(
    messages: Sequence[ChatMessage],
    system_prompt: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
    template: Optional[dict] = {},  # Model specific template
) -> str:
    # (end tokens, structure, etc)
    # @TODO Pass these in from UI model_configs.json (values found in config.json of HF model card)
    BOS = template["BOS"] or ""
    EOS = template["EOS"] or ""
    B_INST = template["B_INST"] or ""
    E_INST = template["E_INST"] or ""
    B_SYS = template["B_SYS"] or ""
    E_SYS = template["E_SYS"] or ""

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


# Methods


# Return a model trained for instruction and RAG, a High level llama-cpp-python object wrapped in LlamaIndex class
def load_text_retrieval_model(
    options: dict,
    callback_manager: CallbackManager = None,  # Optional, debugging
):
    # @TODO We hardcode this until we support user specified model/file. Determine path from HF cache.
    PATH_TO_RETRIEVAL_MODEL = common.app_path(
        os.path.join(common.TEXT_MODELS_CACHE_DIR, "zephyr-7b-beta.Q4_K_M.gguf")
    )
    if not os.path.isfile(PATH_TO_RETRIEVAL_MODEL):
        print(f"{common.PRNT_API} No embedding model exists.")
        # @TODO Will need to await downloading model here if none exists
        raise Exception("No embedding model exists")
    return LlamaCPP(
        # Provide a url to download a model from
        model_url=None,
        # Or, you can set the path to a pre-downloaded model instead of model_url
        model_path=PATH_TO_RETRIEVAL_MODEL,
        # Both max_new_tokens and temperature will override their generate_kwargs counterparts
        max_new_tokens=options["max_tokens"],
        temperature=options["temperature"] or 0,
        # We set this lower to allow for some wiggle room.
        # Note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        # @TODO this can be fixed since we know the model
        context_window=options["n_ctx"],
        # kwargs to pass to __call__()
        generate_kwargs=options["generate_kwargs"],
        # kwargs to pass to __init__()
        model_kwargs=options["model_kwargs"],
        # Transform inputs into model specific format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        callback_manager=callback_manager,
        verbose=True,
    )


# High level llama-cpp-python object wrapped in LlamaIndex class
# https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp/?h=llamacpp
def load_text_model(
    path_to_model: str,
    responseMode: CHAT_MODES,
    init_settings: classes.LoadTextInferenceInit,  # init settings
    gen_settings: classes.LoadTextInferenceCall,  # generation settings
    callback_manager: CallbackManager = None,  # Optional, debugging
):
    n_ctx = init_settings.n_ctx or classes.DEFAULT_MIN_CONTEXT_WINDOW
    if n_ctx <= 0:
        n_ctx = classes.DEFAULT_MIN_CONTEXT_WINDOW
    seed = init_settings.seed
    temperature = gen_settings.temperature
    m_tokens = gen_settings.max_tokens
    max_tokens = common.calc_max_tokens(m_tokens, n_ctx, mode=responseMode)
    n_threads = init_settings.n_threads  # None means auto calc
    if n_threads == -1:
        n_threads = None

    generate_kwargs = {
        "stream": gen_settings.stream,
        "stop": gen_settings.stop,  # !Never use an empty string like [""]
        "echo": gen_settings.echo,
        "model": gen_settings.model,
        "mirostat_tau": gen_settings.mirostat_tau,
        "tfs_z": gen_settings.tfs_z,
        "top_k": gen_settings.top_k,
        "top_p": gen_settings.top_p,
        "min_p": gen_settings.min_p,
        "repeat_penalty": gen_settings.repeat_penalty,
        "presence_penalty": gen_settings.presence_penalty,
        "frequency_penalty": gen_settings.frequency_penalty,
        "temperature": temperature,
        "seed": seed,
        "grammar": gen_settings.grammar,
        "max_tokens": max_tokens,
    }

    model_kwargs = {
        "n_gpu_layers": init_settings.n_gpu_layers,
        "use_mmap": init_settings.use_mmap,
        "use_mlock": init_settings.use_mlock,
        "f16_kv": init_settings.f16_kv,
        "seed": seed,
        "n_ctx": n_ctx,
        "n_batch": init_settings.n_batch,
        "n_threads": n_threads,
        "offload_kqv": init_settings.offload_kqv,
        # "chat_format": "llama-2",  # @TODO Load from model_configs.chat_format
        "torch_dtype": "auto",  # if using CUDA (reduces memory usage)
        # "load_in_8bit": True,
    }

    # @TODO Can we update these without needing to unload model?
    # From: https://docs.llamaindex.ai/en/stable/examples/llm/llama_2_llama_cpp.html
    llm = LlamaCPP(
        # Provide a url to download a model from
        model_url=None,
        # Or, you can set the path to a pre-downloaded model instead of model_url
        model_path=path_to_model,
        # Both max_new_tokens and temperature will override their generate_kwargs counterparts
        max_new_tokens=max_tokens,
        temperature=temperature,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room.
        # Note, this sets n_ctx in the model_kwargs below, so you don't need to pass it there.
        context_window=n_ctx,
        # kwargs to pass to __call__()
        generate_kwargs=generate_kwargs,
        # kwargs to pass to __init__()
        model_kwargs=model_kwargs,
        # Transform inputs into model specific format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm


# Remove from memory
def unload_text_model(llm):
    # Python garbage collector should cleanup if no ref to obj exists
    # https://github.com/abetlen/llama-cpp-python/issues/302
    del llm


def token_streamer(token_generator):
    # @TODO We may need to do some token parsing here...multi-byte encoding can cut off emoji/khanji chars.
    # result = "" # accumulate a final response to be encoded in utf-8 in entirety
    try:
        for token in token_generator:
            payload = {"event": "GENERATING_TOKENS", "data": f"{token}"}
            # print(token, end="", flush=True)
            yield json.dumps(payload)
    except (ValueError, UnicodeEncodeError, Exception) as e:
        msg = f"Error streaming tokens: {e}"
        # print(msg)
        raise Exception(msg)


# Perform a streamed (synchronous) text completion on a prompt with trained data only
def text_stream_completion(
    prompt: str,
    system_message: str,
    message_format: str,
    app,
    options,
):
    sys_message = system_message or ""
    llm: LlamaCPP = app.state.llm
    if llm == None:
        raise Exception("No Ai loaded.")

    # Format to model spec, construct a message with system message and prompt
    message = completion_to_prompt(prompt, sys_message, message_format)

    print(f"{common.PRNT_API} Text Stream Completion: {message}", flush=True)

    # Stream response
    token_generator = llm.stream_complete(message, formatted=True, kwargs=options)
    for token in token_generator:
        # print(token.delta, end="", flush=True)
        payload = {"event": "GENERATING_TOKENS", "data": f"{token.delta}"}
        yield json.dumps(payload)


# Perform a non-streamed (synchronous) text completion on a prompt with trained data only
def text_completion(
    prompt: str,
    system_message: str,
    message_format: str,
    app,
    options,
):
    sys_message = system_message or ""
    llm: LlamaCPP = app.state.llm
    if llm == None:
        raise Exception("No Ai loaded.")

    # Format to model spec, construct a message with system message and prompt
    message = completion_to_prompt(prompt, sys_message, message_format)

    print(f"{common.PRNT_API} Text Non Stream Completion: {message}", flush=True)

    # Get response
    res = llm.complete(message, formatted=True, kwargs=options)
    return res


# Perform a normal text chat conversation
def text_chat(
    messages: Sequence[str],
    system_message: str,
    message_format: str,
    app,
    options,
):
    llm: LlamaCPP = app.state.llm
    if llm == None:
        raise Exception("No Ai loaded.")

    sys_message = system_message or ""
    formatted_messages = messages

    if message_format:
        # Manually format to model spec, inject system message and prompt into query
        formatted_messages = messages_to_prompt(messages, sys_message)

    # Stream response
    token_generator = llm.stream_chat(formatted_messages, kwargs=options)
    for token in token_generator:
        payload = {"event": "GENERATING_TOKENS", "data": f"{token.delta}"}
        yield json.dumps(payload)
