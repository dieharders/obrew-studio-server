"""
Query utilities for RAG (Retrieval Augmented Generation).

This module provides prompt templates and helper functions for building
RAG queries. The actual querying is done via ChromaDB directly.
"""

from typing import Optional
from core import common


# Build prompts

SIMPLE_RAG_PROMPT_TEMPLATE = (
    "We have provided context information below.\n"
    "```\n"
    "{context_str}"
    "\n```\n"
    "Given this information, please answer the question: {user_prompt}\n"
)

REFINE_TEMPLATE = (
    "The original question is as follows: {user_prompt}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n```\n{context_str}\n```\nUsing both the new"
    " context and your own knowledge, update or repeat the existing answer.\n"
)


def build_qa_prompt(
    template: Optional[str] = None,
    query: str = "",
    context: str = "",
) -> str:
    """Build a QA prompt with context and query substituted."""
    prompt_template = template or SIMPLE_RAG_PROMPT_TEMPLATE
    prompt = prompt_template.replace("{context_str}", context)
    prompt = prompt.replace("{user_prompt}", query)
    return prompt


def build_refine_prompt(
    query: str = "",
    existing_answer: str = "",
    context: str = "",
) -> str:
    """Build a refine prompt for improving existing answers."""
    prompt = REFINE_TEMPLATE.replace("{user_prompt}", query)
    prompt = prompt.replace("{existing_answer}", existing_answer)
    prompt = prompt.replace("{context_str}", context)
    return prompt


def format_context_from_results(results: dict) -> str:
    """
    Format ChromaDB query results into a context string for LLM.

    Args:
        results: ChromaDB query results

    Returns:
        Formatted context string
    """
    documents = results.get("documents", [[]])[0]
    context_parts = []
    for i, doc in enumerate(documents):
        context_parts.append(f"[{i+1}] {doc}")
    return "\n\n".join(context_parts)


def log_query_results(results: dict):
    """Log query results for debugging."""
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (chunk_id, doc, distance) in enumerate(zip(ids, documents, distances)):
        score = 1 - distance  # Convert distance to similarity score
        print(
            f"{common.PRNT_EMBED} chunk id::{chunk_id} | score={score:.4f}\ntext=\n{doc[:200]}...",
            flush=True,
        )
