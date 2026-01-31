from typing import Literal, Optional
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Perform file operations. Use this tool to create, edit, delete, move, copy, or rename files in the filesystem."""

    operation: Literal["create", "edit", "delete", "move", "copy", "rename"] = Field(
        ...,
        description="The file operation to perform.",
        options=["create", "edit", "delete", "move", "copy", "rename"],
    )
    path: str = Field(
        ...,
        description="The path to the file to operate on.",
    )
    content: Optional[str] = Field(
        default=None,
        description="The content for create/edit operations.",
    )
    destination: Optional[str] = Field(
        default=None,
        description="The destination path for move/copy/rename operations.",
    )
    create_dirs: bool = Field(
        default=True,
        description="Whether to create parent directories if they don't exist.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "operation": "create",
                    "path": "/documents/report.md",
                    "content": "# Report\n\nThis is the report content.",
                    "create_dirs": True,
                },
                {
                    "operation": "move",
                    "path": "/old/location/file.txt",
                    "destination": "/new/location/file.txt",
                },
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    operation = kwargs.get("operation")
    path = kwargs.get("path")
    content = kwargs.get("content")
    destination = kwargs.get("destination")
    create_dirs = kwargs.get("create_dirs", True)

    if not operation:
        raise ValueError("An operation type is required.")

    if not path:
        raise ValueError("A file path is required.")

    # Validate operation-specific requirements
    if operation in ["create", "edit"] and content is None:
        raise ValueError(f"Content is required for '{operation}' operation.")

    if operation in ["move", "copy", "rename"] and not destination:
        raise ValueError(f"Destination path is required for '{operation}' operation.")

    # Build the operation details
    result = {
        "operation": operation,
        "path": path,
        "create_dirs": create_dirs,
        "status": "ready",
    }

    if content is not None:
        result["content"] = content
        result["content_length"] = len(content)

    if destination:
        result["destination"] = destination

    return result
