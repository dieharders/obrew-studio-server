"""
Tool: workflow_plan
Plan a workflow by listing the node types needed, in execution order.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any


VALID_NODE_TYPES = [
    "begin", "end", "text", "number", "file", "data", "api", "workflow",
    "context", "context-search", "ai-process", "data-transform", "file-sync",
    "email-notify", "calendar", "meeting", "compliance-check", "email-insights",
    "json-output", "markdown-output", "txt-output", "csv-output", "xml-output",
    "switch", "comparison", "delay", "stop", "interval",
]


class Params(BaseModel):
    """Plan a workflow by listing the node types needed, in execution order.
    Always start with 'begin' and end with 'end'.
    Place source nodes (text, file, context, context-search) before the action that uses them."""
    nodes: List[str] = Field(
        ...,
        description="Ordered list of node type IDs",
        min_length=2,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "nodes": ["begin", "text", "ai-process", "email-notify", "end"]
                }
            ]
        }
    }


async def main(**kwargs) -> Dict[str, Any]:
    nodes_raw = kwargs.get("nodes", [])

    # Filter to valid types only
    validated = [n for n in nodes_raw if isinstance(n, str) and n in VALID_NODE_TYPES]

    # Ensure begin/end bookends
    if not validated or validated[0] != "begin":
        validated.insert(0, "begin")
    if validated[-1] != "end":
        validated.append("end")

    return {"nodes": validated, "success": True}
