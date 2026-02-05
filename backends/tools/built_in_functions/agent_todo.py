from typing import List, Literal
from pydantic import BaseModel, Field


# @TODO This is a stub. This tool needs to take a prompt and break it down in order to output a todo. This list contains "tasks" which are the arguments used to call the "agent_task" tool. The harness calling this tool will use each task's value to call the "agent_task" tool until it has completed the last task in the todo.
class Params(BaseModel):
    """Execute a list of tasks sequentially. Use this tool to process multiple related tasks in order, optionally stopping on errors."""

    tasks: List[str] = Field(
        ...,
        description="A list of task descriptions to execute in order.",
    )
    stop_on_error: bool = Field(
        default=True,
        description="Whether to stop execution if a task fails.",
    )
    execution_mode: Literal["sequential", "summary"] = Field(
        default="sequential",
        description="How to process tasks: 'sequential' executes each task, 'summary' provides an overview.",
        options=["sequential", "summary"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "tasks": [
                        "Review the document structure",
                        "Identify key themes",
                        "Generate a summary",
                    ],
                    "stop_on_error": True,
                    "execution_mode": "sequential",
                }
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    tasks = kwargs.get("tasks", [])
    stop_on_error = kwargs.get("stop_on_error", True)
    execution_mode = kwargs.get("execution_mode", "sequential")

    if not tasks:
        raise ValueError("At least one task is required.")

    if not isinstance(tasks, list):
        raise ValueError("Tasks must be provided as a list.")

    # Build the todo list structure
    todo_list = []
    for i, task in enumerate(tasks):
        todo_list.append(
            {
                "id": i + 1,
                "task": task,
                "status": "pending",
            }
        )

    return {
        "todo_list": todo_list,
        "total_tasks": len(tasks),
        "stop_on_error": stop_on_error,
        "execution_mode": execution_mode,
        "status": "ready",
    }
