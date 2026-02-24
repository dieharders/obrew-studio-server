from typing import List
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Generate todo linked-lists of step-by-step tasks that accomplish a given goal."""

    todo_tasks: List[str] = Field(
        ...,
        description="A linked-list of task descriptions to be executed in order.",
    )
    todo_instructions: List[str] = Field(
        ...,
        description="A linked-list of arguments passed to a 'task' tool function",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "todo_tasks": [
                        "Review the document structure",
                        "Identify key themes",
                        "Generate a summary",
                    ],
                    "todo_instructions": [
                        "Read through the entire document and note how it is organized, including sections, headings, sub-sections, and any visual or structural elements like tables or lists",
                        "Go through each section and pull out the main topics, recurring ideas, and any patterns or connections between them",
                        "Using the structure and themes identified, compose a clear and concise summary that captures the document's purpose, key findings, and conclusions",
                    ],
                }
            ]
        }
    }


# The harness calling this tool will use each task's value to call
# the "agent_task" tool until it has completed the last task in the todo.
#
# It is expected that the consumer passes the prompt (query) in this format:
#
# ## Goal
# <The user's goal or objective to break down into tasks>
#
# ## Context
# <context>
# <Any relevant state, history, or background info to inform task planning>
# </context>
#
# NOTE: Context is wrapped in <context></context> XML tags because it may
# contain prior LLM output or user-provided text that could be adversarial.
async def main(**kwargs: Params) -> dict:
    todo_tasks = kwargs.get("todo_tasks", [])
    todo_instructions = kwargs.get("todo_instructions", [])

    if not todo_tasks:
        raise ValueError("todo_tasks was not generated.")
    if not isinstance(todo_tasks, list):
        raise ValueError("todo_tasks must be a list.")

    if not todo_instructions:
        raise ValueError("todo_instructions was not generated.")
    if not isinstance(todo_instructions, list):
        raise ValueError("todo_instructions must be a list.")

    # Build the structured todo list by pairing each task with its instruction
    todo_list = []
    for i, task in enumerate(todo_tasks):
        todo_list.append(
            {
                "id": i + 1,
                "task": task,
                "instruction": (
                    todo_instructions[i] if i < len(todo_instructions) else ""
                ),
            }
        )

    return {
        "todo_list": todo_list,
        "total_tasks": len(todo_list),
    }
