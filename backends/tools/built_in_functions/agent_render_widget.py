from typing import Literal, Optional, List
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Generate data for rendering an interactive widget. Use this tool to create structured data for UI widgets like email previews."""

    widget_type: Literal["email"] = Field(
        ...,
        description="The type of widget to render. Currently supports 'email'.",
        options=["email"],
    )
    data: dict = Field(
        ...,
        description="The data to populate the widget. Structure depends on widget_type.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "widget_type": "email",
                    "data": {
                        "to": ["user@example.com"],
                        "cc": [],
                        "bcc": [],
                        "subject": "Meeting Follow-up",
                        "body": "Thank you for attending the meeting...",
                        "attachments": [],
                    },
                }
            ]
        }
    }


class EmailData(BaseModel):
    """Schema for email widget data."""
    to: List[str] = Field(..., description="List of recipient email addresses.")
    cc: Optional[List[str]] = Field(default=[], description="List of CC email addresses.")
    bcc: Optional[List[str]] = Field(default=[], description="List of BCC email addresses.")
    subject: str = Field(..., description="Email subject line.")
    body: str = Field(..., description="Email body content.")
    attachments: Optional[List[str]] = Field(default=[], description="List of attachment file paths.")


def validate_email_data(data: dict) -> dict:
    """Validate and normalize email widget data."""
    required_fields = ["to", "subject", "body"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Email widget requires '{field}' field.")

    # Ensure 'to' is a list
    if isinstance(data.get("to"), str):
        data["to"] = [data["to"]]

    # Set defaults for optional fields
    data.setdefault("cc", [])
    data.setdefault("bcc", [])
    data.setdefault("attachments", [])

    return data


async def main(**kwargs: Params) -> dict:
    widget_type = kwargs.get("widget_type")
    data = kwargs.get("data", {})

    if not widget_type:
        raise ValueError("A widget_type is required.")

    if not data:
        raise ValueError("Widget data is required.")

    # Validate data based on widget type
    validators = {
        "email": validate_email_data,
    }

    if widget_type not in validators:
        raise ValueError(f"Unsupported widget type: {widget_type}")

    validated_data = validators[widget_type](data)

    return {
        "widget_type": widget_type,
        "data": validated_data,
        "status": "ready",
    }
