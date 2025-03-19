from typing import Any
from inference.classes import RagTemplateData
from core import common, classes
from llama_index.core import VectorStoreIndex, PromptTemplate
from llama_index.core.response_synthesizers import ResponseMode


# Build prompts

SIMPLE_RAG_PROMPT_TEMPLATE = (
    "We have provided context information below.\n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)


# Used when no good response is returned and we want to further "handle" the answer before its delivered to user.
def build_refine_prompt() -> PromptTemplate:
    # @TODO Hardcoded for now, Set this from passed args in request
    refine_template_str = (
        "The original question is as follows: {query_str}\nWe have provided an"
        " existing answer: {existing_answer}\nWe have the opportunity to refine"
        " the existing answer (only if needed) with some more context"
        " below.\n------------\n{context_str}\n------------\nUsing both the new"
        " context and your own knowledge, update or repeat the existing answer.\n"
    )
    return PromptTemplate(refine_template_str)


def build_qa_prompt(template, prompt_type):
    p_type = prompt_type or "custom_default"
    if template:
        return PromptTemplate(template=template, prompt_type=p_type)
    else:
        return PromptTemplate(template=SIMPLE_RAG_PROMPT_TEMPLATE, prompt_type=p_type)


# Query Private Data (RAG)
def query_embedding(
    llm: Any,
    query: str,
    prompt_template: RagTemplateData,
    index: VectorStoreIndex,
    options: classes.ContextRetrievalOptions,
    streaming: bool,
):
    print(
        f"{common.PRNT_EMBED} Query Data:\n{prompt_template.text}\n{prompt_template.type}",
        flush=True,
    )

    # Construct a prompt from a template
    custom_qa_prompt = build_qa_prompt(
        template=prompt_template.text, prompt_type=prompt_template.type
    )

    # Call query() in query mode
    print(f"{common.PRNT_EMBED} Query prompt:\n{custom_qa_prompt}", flush=True)
    try:
        query_engine = index.as_query_engine(
            llm=llm,
            streaming=streaming,
            # summary_template=summary_template,
            # simple_template=simple_template,
            text_qa_template=custom_qa_prompt,
            refine_template=build_refine_prompt(),
            similarity_top_k=options["similarity_top_k"] or 1,
            response_mode=options["response_mode"] or ResponseMode.COMPACT,
        )
        # @TODO in chat mode
        # chat_engine = index.as_chat_engine(...)

        streaming_response = query_engine.query(query)
    except Exception as err:
        raise Exception(f"Query engine failed to return result. {err}")
    # Log probability scores (logits) for each chunk
    for node in streaming_response.source_nodes:
        print(
            f"{common.PRNT_EMBED} chunk id::{node.id_} | score={node.score}\ntext=\n{node.text}",
            flush=True,
        )
    return streaming_response
