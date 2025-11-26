"""
Response evaluation utilities.

NOTE: These functions are stubs and need implementation.
The previous implementation used llama-index's FaithfulnessEvaluator which
required an LLM connection. A new implementation could use a local LLM
to evaluate response faithfulness.
"""


def contributing_references(response: dict, eval_result: dict = None):
    """
    Determine which nodes contributed to the answer.

    Args:
        response: Response containing source_nodes
        eval_result: Optional evaluation result

    Returns:
        Dict with reference information
    """
    # Stub implementation
    source_nodes = response.get("source_nodes", [])
    num_source_nodes = len(source_nodes)
    print(f"[embedding api] Number of source nodes: {num_source_nodes}", flush=True)
    return {
        "num_refs": num_source_nodes,
    }


def verify_response(response: dict, query: str = ""):
    """
    Verify whether a response is faithful to the contexts.

    NOTE: This is a stub. Needs implementation with local LLM.

    Args:
        response: The response to verify
        query: The original query
    """
    print("[embedding api] Response verification not implemented.", flush=True)
    print("[embedding api] Skipping faithfulness check.", flush=True)


def evaluate_response(response: dict, query: str = ""):
    """
    Evaluate whether a response is faithful to the query.

    NOTE: This is a stub. Needs implementation with local LLM.

    Args:
        response: The response to evaluate
        query: The original query
    """
    print("[embedding api] Response evaluation not implemented.", flush=True)
    print("[embedding api] Skipping evaluation check.", flush=True)
