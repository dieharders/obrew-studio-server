from enum import Enum


class RESPONSE_SYNTHESIS_DESCRIPTIONS(str, Enum):
    REFINE = "Refine is an iterative way of generating a response.\nWe first use the context in the first node, along with the query, to generate an initial answer. We then pass this answer, the query, and the context of the second node as input into a “refine prompt” to generate a refined answer. We refine through N-1 nodes, where N is the total number of nodes."
    COMPACT = "Compact and refine mode first combine text chunks into larger consolidated chunks that more fully utilize the available context window, then refine answers across them. This mode is faster than refine since we make fewer calls to the LLM."
    SIMPLE_SUMMARIZE = "Merge all text chunks into one, and make a LLM call. This will fail if the merged text chunk exceeds the context window size."
    TREE_SUMMARIZE = "Build a tree index over the set of candidate nodes, with a summary prompt seeded with the query.\nThe tree is built in a bottoms-up fashion, and in the end the root node is returned as the response."
    NO_TEXT = (
        "Return the retrieved context nodes, without synthesizing a final response."
    )
    CONTEXT_ONLY = "Returns a concatenated string of all text chunks."
    ACCUMULATE = (
        "Synthesize a response for each text chunk, and then return the concatenation."
    )
    COMPACT_ACCUMULATE = "Compact and accumulate mode first combine text chunks into larger consolidated chunks that more fully utilize the available context window, then accumulate answers for each of them and finally return the concatenation.\nThis mode is faster than accumulate since we make fewer calls to the LLM."


class RESPONSE_SYNTHESIS_MODES(str, Enum):
    """Modes of synthesizing responses using context taken from embeddings."""

    REFINE = "refine"
    COMPACT = "compact"
    SIMPLE_SUMMARIZE = "simple_summarize"
    TREE_SUMMARIZE = "tree_summarize"
    NO_TEXT = "no_text"
    CONTEXT_ONLY = "context_only"
    ACCUMULATE = "accumulate"
    COMPACT_ACCUMULATE = "compact_accumulate"
