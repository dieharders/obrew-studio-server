import json
from typing import Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Analyze Teams chat messages to produce summaries, extract key points, or find all shared links. Use this tool when the user wants to understand a Teams chat conversation, get a summary of discussions, identify action items, or collect links shared in the chat.

    Fields by action:
    - 'summarize': chat_messages, chat_name
    - 'key_points': chat_messages, chat_name
    - 'extract_links': chat_messages, chat_name
    """

    action: Optional[str] = Field(
        default="summarize",
        description="The type of analysis to perform. One of: 'summarize' (concise summary of the chat), 'key_points' (extract decisions, action items, and key discussion points), 'extract_links' (find all URLs shared in the chat with context). Default: 'summarize'.",
    )
    chat_messages: Optional[str] = Field(
        default="[]",
        description="JSON array of chat message objects. Each message should have: sender (string), content (string), timestamp (string). These are the messages to analyze.",
    )
    chat_name: Optional[str] = Field(
        default="",
        description="The name or topic of the Teams chat being analyzed. Used for context in the analysis output.",
    )
    summary: Optional[str] = Field(
        default="",
        description="For 'summarize' action: A concise summary of the chat conversation covering main topics, decisions made, and overall tone. 2-4 paragraphs.",
    )
    key_points: Optional[str] = Field(
        default="[]",
        description="For 'key_points' action: JSON array of strings, each being a key point, decision, or action item extracted from the conversation. Be specific and actionable.",
    )
    links: Optional[str] = Field(
        default="[]",
        description="For 'extract_links' action: JSON array of objects with 'url' (the link) and 'context' (brief description of why/how the link was shared). Extract ALL URLs from the messages.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "action": "summarize",
                    "chat_messages": '[{"sender": "Alice", "content": "Has everyone reviewed the Q4 budget proposal?", "timestamp": "2025-01-15T09:00:00"}, {"sender": "Bob", "content": "Yes, I think we should increase the marketing allocation by 15%", "timestamp": "2025-01-15T09:05:00"}]',
                    "chat_name": "Project Alpha Team",
                    "summary": "The team discussed the Q4 budget proposal. Bob suggested a 15% increase in marketing allocation. The conversation focused on budget planning for the upcoming quarter.",
                    "key_points": "[]",
                    "links": "[]",
                },
                {
                    "action": "key_points",
                    "chat_messages": '[{"sender": "Alice", "content": "We need to finalize the vendor contract by Friday", "timestamp": "2025-01-15T10:00:00"}, {"sender": "Charlie", "content": "I will send the updated proposal to legal today", "timestamp": "2025-01-15T10:02:00"}]',
                    "chat_name": "Vendor Negotiations",
                    "summary": "",
                    "key_points": '["Vendor contract must be finalized by Friday", "Charlie will send updated proposal to legal today"]',
                    "links": "[]",
                },
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    """Return structured chat analysis data for the Teams Chat Insights component.

    The LLM fills in the appropriate output fields based on the action type.
    The frontend uses these to render summaries, key points, or link lists.
    """
    action = kwargs.get("action", "summarize") or "summarize"
    chat_messages_json = kwargs.get("chat_messages", "[]") or "[]"
    chat_name = kwargs.get("chat_name", "") or ""
    summary = kwargs.get("summary", "") or ""
    key_points_json = kwargs.get("key_points", "[]") or "[]"
    links_json = kwargs.get("links", "[]") or "[]"

    # Parse chat messages to compute stats
    try:
        messages = json.loads(chat_messages_json) if isinstance(chat_messages_json, str) else chat_messages_json
    except (json.JSONDecodeError, TypeError):
        messages = []

    # Compute basic stats from messages
    participant_counts = {}
    command_count = 0
    for msg in messages:
        if isinstance(msg, dict):
            sender = msg.get("sender", "Unknown")
            participant_counts[sender] = participant_counts.get(sender, 0) + 1
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip().startswith("/"):
                command_count += 1

    # Parse key points
    try:
        key_points_list = json.loads(key_points_json) if isinstance(key_points_json, str) else key_points_json
    except (json.JSONDecodeError, TypeError):
        key_points_list = []

    # Parse links
    try:
        links_list = json.loads(links_json) if isinstance(links_json, str) else links_json
    except (json.JSONDecodeError, TypeError):
        links_list = []

    return {
        "action": action,
        "chatName": chat_name,
        "summary": summary if summary else None,
        "keyPoints": key_points_list if key_points_list else None,
        "links": links_list if links_list else None,
        "stats": {
            "messageCount": len(messages),
            "participantCount": len(participant_counts),
            "commandCount": command_count,
        },
    }
