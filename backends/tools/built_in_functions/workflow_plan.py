from pydantic import BaseModel, Field


class Params(BaseModel):
    """Plan a workflow step-by-step. Given an instruction and accumulated context, generate a clarifying question with multiple choices. Returns an empty question when enough information has been gathered."""

    instruction: str = Field(
        ...,
        description="The user's original request or goal for the workflow.",
    )
    current_context: str = Field(
        default="",
        description="Accumulated answers and decisions gathered so far.",
    )
    phase: str = Field(
        default="workflow",
        description="Current planning phase: 'workflow' for name/description/goal, 'node' for choosing which node to add next, 'settings' for configuring a specific node's settings, 'connections' for determining how to connect a node.",
        options=["workflow", "node", "settings", "connections"],
    )
    graph_state: str = Field(
        default="",
        description="Text representation of the current workflow graph state.",
    )
    available_nodes: str = Field(
        default="",
        description="Comma-separated list of node types available to choose from.",
    )
    node_type: str = Field(
        default="",
        description="The current node type being configured (for settings/connections phases).",
    )
    setting_name: str = Field(
        default="",
        description="The specific setting being configured (for settings phase).",
    )
    setting_options: str = Field(
        default="",
        description="Available options for the current setting, comma-separated.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "instruction": "Create a workflow that processes documents and sends email summaries",
                    "current_context": "Purpose: Document processing\nFrequency: Daily",
                    "phase": "workflow",
                    "graph_state": "",
                    "available_nodes": "",
                    "node_type": "",
                    "setting_name": "",
                    "setting_options": "",
                }
            ]
        }
    }


RESPONSE_FORMAT = """Respond in EXACTLY this format:
QUESTION: [Your clarifying question here]
A: [First choice] (RECOMMENDED)
B: [Second choice]
C: [Third choice]
D: [Fourth choice]

Or if you have enough information to proceed, respond with exactly:
READY"""

PHASE_PROMPTS = {
    "workflow": (
        "You are a workflow planning assistant helping a user define their automation workflow. "
        "Your job is to ask ONE clarifying question at a time to refine the workflow's name, description, and goal. "
        "Focus on understanding: what the workflow does, what inputs it needs, what outputs it produces, and when/how often it runs. "
        "Mark your recommended choice with (RECOMMENDED). "
        "When you have a clear understanding of the workflow's purpose, scope, and goal, output READY."
    ),
    "node": (
        "You are helping build a workflow by adding nodes one at a time. "
        "Given the current graph state and workflow goal, suggest the NEXT node to add. "
        "Present your suggestion as a question with the node type choices from the available list. "
        "Consider the logical execution order: sources and inputs first, then processing, then outputs. "
        "Mark your recommended choice with (RECOMMENDED). "
        "When the workflow has all necessary nodes to accomplish the goal, output READY."
    ),
    "settings": (
        "You are configuring a workflow node's settings. "
        "For the current setting, suggest appropriate values as choices. "
        "Consider the workflow's goal and the node's purpose when recommending values. "
        "Mark your recommended choice with (RECOMMENDED). "
        "When the setting value is clear from context, output READY with the recommended value."
    ),
    "connections": (
        "You are connecting workflow nodes together. "
        "Given the current graph state and the node that needs connections, suggest which existing node(s) and handles to connect to. "
        "Consider data type compatibility and logical data flow. "
        "Mark your recommended choice with (RECOMMENDED). "
        "When connections are determined, output READY."
    ),
}


async def main(**kwargs) -> dict:
    instruction = kwargs.get("instruction", "")
    current_context = kwargs.get("current_context", "")
    phase = kwargs.get("phase", "workflow")
    graph_state = kwargs.get("graph_state", "")
    available_nodes = kwargs.get("available_nodes", "")
    node_type = kwargs.get("node_type", "")
    setting_name = kwargs.get("setting_name", "")
    setting_options = kwargs.get("setting_options", "")

    if not instruction:
        raise ValueError("An instruction is required.")

    # Build the system prompt for this phase
    system_prompt = PHASE_PROMPTS.get(phase, PHASE_PROMPTS["workflow"])

    # Build the user prompt with all available context
    parts = [f"Goal: {instruction}"]

    if current_context:
        parts.append(f"\nDecisions so far:\n{current_context}")

    if graph_state:
        parts.append(f"\nCurrent workflow graph:\n{graph_state}")

    if phase == "node" and available_nodes:
        parts.append(f"\nAvailable node types to choose from: {available_nodes}")

    if phase == "settings":
        if node_type:
            parts.append(f"\nConfiguring node type: {node_type}")
        if setting_name:
            parts.append(f"\nSetting to configure: {setting_name}")
        if setting_options:
            parts.append(f"\nAvailable options: {setting_options}")

    if phase == "connections" and node_type:
        parts.append(f"\nNode to connect: {node_type}")

    parts.append(f"\n{RESPONSE_FORMAT}")

    full_prompt = "\n".join(parts)

    return {
        "plan_prompt": full_prompt,
        "system": system_prompt,
        "phase": phase,
        "instruction": instruction,
        "status": "ready",
    }
