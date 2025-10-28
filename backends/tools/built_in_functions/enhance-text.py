from pydantic import BaseModel, Field
from core.classes import FastAPIApp
from inference.helpers import read_event_data


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Ask an expert writer to enhance the provided context."""

    context: str = Field(
        ...,
        description="The text to enhance, transform or critique.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"context": "This is an example context that needs enhancing."},
            ]
        }
    }


# JSON schema for constraining the LLM output
ENHANCEMENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "enhancements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["grammar", "style", "clarity", "format", "insight"],
                    },
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "suggestion": {"type": "string"},
                    "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                },
                "required": [
                    "type",
                    "title",
                    "description",
                    "suggestion",
                    "confidence",
                ],
            },
            "minItems": 2,
            "maxItems": 5,
        }
    },
    "required": ["enhancements"],
}


async def main(**kwargs: Params) -> str:
    """
    Analyze text and return enhancement suggestions in JSON format.
    The output is constrained by JSON schema to ensure valid structure.

    IMPORTANT: This tool should be called with tool_response_type="result"
    to return the raw JSON directly without LLM interpretation.
    """
    app: FastAPIApp = kwargs.get("app")
    context = kwargs.get("context")

    if not context:
        raise ValueError("Context text is required for enhancement.")

    # Use the app's LLM instance
    llm = app.state.llm

    # Simplified system message - no need to explain JSON format since it's constrained
    system_message = """You are an expert writing coach and editor. Analyze the provided text and suggest 2-5 specific, actionable enhancements.

For each enhancement:
- type: Choose the most appropriate category (grammar, style, clarity, format, or insight)
- title: A brief, descriptive title
- description: Explain what needs improvement and why
- suggestion: Provide the improved version of the text
- confidence: Rate your confidence in this enhancement (0-100)

Focus on the most impactful improvements."""

    # Simplified prompt
    prompt = f"""Analyze the following text and provide enhancement suggestions:

{context}"""

    # Call the LLM with JSON schema constraint
    response = await llm.text_completion(
        request=kwargs.get("request"),
        prompt=prompt,
        system_message=system_message,
        constrain_json_output=ENHANCEMENT_OUTPUT_SCHEMA,
    )

    # Collect the complete response
    content = [item async for item in response]
    data = read_event_data(content)

    # Extract the text response - should already be valid JSON due to schema constraint
    result_text = data.get("text", "")

    if not result_text:
        raise Exception("No enhancement response generated from LLM.")

    # Return the JSON text directly - constrain_json_output ensures it's valid
    return result_text
