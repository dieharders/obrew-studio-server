from typing import Optional, List
from pydantic import BaseModel, Field


class LinkItem(BaseModel):
    url: str = Field(default="", description="The URL found in the chat messages.")
    context: str = Field(
        default="",
        description="Brief description of why or how the link was shared.",
    )


class Params(BaseModel):
    """Analyze Teams chat messages to produce summaries, extract key points, or find all shared links. Use this tool when the user wants to understand a Teams chat conversation, get a summary of discussions, identify action items, or collect links shared in the chat.

    Output the analysis result in the appropriate field based on the action requested in the prompt.
    - For 'summarize': fill the summary field
    - For 'key_points': fill the key_points field
    - For 'extract_links': fill the links field
    """

    action: Optional[str] = Field(
        default="summarize",
        description="The type of analysis performed. One of: 'summarize', 'key_points', 'extract_links'.",
    )
    chat_name: Optional[str] = Field(
        default="",
        description="The name or topic of the Teams chat that was analyzed.",
    )
    summary: Optional[str] = Field(
        default="",
        description="For 'summarize' action: A concise summary of the chat conversation covering main topics, decisions made, and overall tone. 2-4 paragraphs.",
    )
    key_points: Optional[List[str]] = Field(
        default=[],
        description="For 'key_points' action: A list of key points, decisions, and action items extracted from the conversation. Be specific and actionable.",
    )
    links: Optional[List[LinkItem]] = Field(
        default=[],
        description="For 'extract_links' action: A list of links found in the chat messages, each with the URL and context for why it was shared.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "summarize",
                    "chat_name": "Project Alpha Team",
                    "summary": "The team discussed the Q4 budget proposal. Bob suggested a 15% increase in marketing allocation. The conversation focused on budget planning for the upcoming quarter.",
                    "key_points": [],
                    "links": [],
                },
                {
                    "action": "key_points",
                    "chat_name": "Vendor Negotiations",
                    "summary": "",
                    "key_points": [
                        "Vendor contract must be finalized by Friday",
                        "Charlie will send updated proposal to legal today",
                    ],
                    "links": [],
                },
            ]
        }
    }


async def main(**kwargs) -> dict:
    """Return structured chat analysis data for the Teams Chat Insights component.

    The LLM fills in the appropriate output fields based on the action type.
    The frontend uses these to render summaries, key points, or link lists.
    Stats are computed on the frontend from the raw messages.
    """
    action = kwargs.get("action", "summarize") or "summarize"
    chat_name = kwargs.get("chat_name", "") or ""
    summary = kwargs.get("summary", "") or ""
    key_points = kwargs.get("key_points", []) or []
    links_raw = kwargs.get("links", []) or []

    # Normalize links — they may arrive as dicts or LinkItem instances
    links = []
    for item in links_raw:
        if isinstance(item, dict):
            links.append(item)
        elif hasattr(item, "model_dump"):
            links.append(item.model_dump())
        else:
            links.append({"url": str(item), "context": ""})

    return {
        "action": action,
        "chatName": chat_name,
        "summary": summary if summary else None,
        "keyPoints": key_points if key_points else None,
        "links": links if links else None,
    }
