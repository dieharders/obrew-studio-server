"""
Tool: workflow_node_detail
Provide details for a single workflow node: a display label and instruction/content text.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any


class Params(BaseModel):
    """Provide details for a single workflow node.
    label: A short display name for this node (2-5 words).
    value: The instruction, prompt, query, or text content for this node. Leave empty if not applicable."""
    label: str = Field(
        ...,
        description="Short display name (2-5 words)",
    )
    value: str = Field(
        default="",
        description="Instruction text, prompt, query, or content",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "label": "Extract Invoice Data",
                    "value": "Extract vendor name, invoice number, line items, and total amount from the document",
                }
            ]
        }
    }


async def main(**kwargs) -> Dict[str, Any]:
    return {
        "label": kwargs.get("label", ""),
        "value": kwargs.get("value", ""),
        "success": True,
    }
