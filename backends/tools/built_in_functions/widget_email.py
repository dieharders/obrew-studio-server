from typing import Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Compose an email by extracting sender, recipient, subject, and body from user intent. Use this tool when the user wants to send, write, or compose an email."""

    sender: Optional[str] = Field(
        default="",
        description="The sender's email address (the 'from' field). Extract if the user mentions 'my email is' or 'from me at'. Leave empty if not specified.",
    )
    to: Optional[str] = Field(
        default="",
        description="Recipient email address(es) - who the email is being sent TO. If multiple, separate with commas. Leave empty if not found in context.",
    )
    subject: Optional[str] = Field(
        default="",
        description="Email subject line summarizing the email purpose. Generate an appropriate subject based on the user's intent.",
    )
    body: Optional[str] = Field(
        default="",
        description="Email body content in plain text. Generate professional and appropriate content based on the user's request and any provided context.",
    )
    cc: Optional[str] = Field(
        default="",
        description="CC email addresses (comma-separated) if mentioned by the user.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sender": "",
                    "to": "john@example.com",
                    "subject": "Meeting Reminder - Tomorrow at 3 PM",
                    "body": "Hi John,\n\nThis is a friendly reminder about our meeting scheduled for tomorrow at 3 PM.\n\nPlease let me know if you need to reschedule.\n\nBest regards",
                    "cc": "",
                },
                {
                    "sender": "alice@mycompany.com",
                    "to": "bob@example.com",
                    "subject": "Quick Question",
                    "body": "Hi Bob,\n\nI had a quick question about the project.\n\nBest,\nAlice",
                    "cc": "",
                },
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    """Return structured email data for widget rendering.

    All fields are optional - the widget UI will handle displaying
    placeholders for any missing data.
    """
    sender = kwargs.get("sender", "")
    to = kwargs.get("to", "")
    subject = kwargs.get("subject", "")
    body = kwargs.get("body", "")
    cc = kwargs.get("cc", "")

    # Parse comma-separated emails into lists (handle empty strings gracefully)
    to_list = [email.strip() for email in to.split(",") if email.strip()] if to else []
    cc_list = [email.strip() for email in cc.split(",") if email.strip()] if cc else []

    return {
        "from": sender,
        "to": to_list,
        "cc": cc_list,
        "subject": subject,
        "body": body,
        "attachments": [],
        "status": "composed",
    }
