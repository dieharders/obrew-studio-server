import json
from typing import Literal
from pydantic import BaseModel, Field
from core.classes import FastAPIApp
from inference.helpers import read_event_data


class Params(BaseModel):
    # Required - A description is needed for prompt injection
    """Ask an expert writer to enhance the provided context."""

    type: Literal["grammar", "style", "clarity", "format", "insight"] = Field(
        ...,
        description="The category of enhancement: `grammar` for spelling/grammar fixes, `style` for writing style improvements, `clarity` for making text clearer, `format` for structure/organization, or `insight` for adding depth or supporting evidence.",
        options=["grammar", "style", "clarity", "format", "insight"],
    )

    title: str = Field(
        ...,
        description="A brief, descriptive title that summarizes the enhancement (e.g., `Fix subject-verb agreement`, `Use active voice`).",
    )

    description: str = Field(
        ...,
        description="A clear explanation of what needs improvement and why this change would make the text better.",
    )

    suggestion: str = Field(
        ...,
        description="The improved version of the text, showing exactly how it should be rewritten.",
    )

    confidence: int = Field(
        ...,
        description="A confidence score from 0 to 100 indicating how certain you are that this enhancement will improve the text.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "grammar",
                    "title": "Subject-verb agreement correction",
                    "description": "The verb `are` should be `is` to match the singular subject `team`.",
                    "suggestion": "The team is working on the project.",
                    "confidence": 95,
                },
                {
                    "type": "clarity",
                    "title": "Simplify complex sentence",
                    "description": "Breaking down the sentence improves readability and makes the main point clearer.",
                    "suggestion": "We should prioritize user feedback. This will help us improve the product faster.",
                    "confidence": 85,
                },
                {
                    "type": "style",
                    "title": "Use active voice",
                    "description": "Active voice is more direct and engaging than passive voice.",
                    "suggestion": "The team completed the analysis yesterday.",
                    "confidence": 90,
                },
                {
                    "type": "insight",
                    "title": "Add supporting evidence",
                    "description": "The claim would be stronger with specific data or examples to support it.",
                    "suggestion": "User engagement increased by `35%` after implementing the new onboarding flow.",
                    "confidence": 80,
                },
                {
                    "type": "format",
                    "title": "Improve paragraph structure",
                    "description": "Breaking the text into shorter paragraphs enhances scannability and readability.",
                    "suggestion": "Start with the main point in the first sentence.\n\nFollow with supporting details in subsequent sentences.\n\nEnd with a clear conclusion or call to action.",
                    "confidence": 88,
                },
            ]
        }
    }


async def main(**kwargs: Params) -> str:
    """
    Analyze text and return enhancement suggestions in JSON format.

    IMPORTANT: This tool should be called with tool_response_type="result"
    to return the raw JSON directly without LLM interpretation. The responseMode
    should be set to "result" to ensure output is constrained to JSON schema.
    """
    app: FastAPIApp = kwargs.get("app")
    type = kwargs.get("type")
    title = kwargs.get("title")
    description = kwargs.get("description")
    suggestion = kwargs.get("suggestion")
    confidence = kwargs.get("confidence")

    if (
        type == None
        or title == None
        or description == None
        or suggestion == None
        or confidence == None
    ):
        raise ValueError("Missing required for enhancement.")

    # Use the app's LLM instance
    # llm = app.state.llm

    # Simplified system message - no need to explain JSON format since it's constrained
    # @TODO Remove or put this in filebuff's handleAnalyzeParagraph()
    #     system_message = """You are an expert writing coach and editor. Analyze the provided text and suggest 2-5 specific, actionable enhancements.

    # For each enhancement:
    # - type: Choose the most appropriate category (grammar, style, clarity, format, or insight)
    # - title: A brief, descriptive title
    # - description: Explain what needs improvement and why
    # - suggestion: Provide the improved version of the text
    # - confidence: Rate your confidence in this enhancement (0-100)

    # Focus on the most impactful improvements."""

    # Simplified prompt @TODO Remove or put this in filebuff's handleAnalyzeParagraph()
    #     prompt = f"""Analyze the following text and provide enhancement suggestions:

    # {context}"""

    # Call the LLM with JSON schema constraint
    # @TODO May not need this, remove
    # response = await llm.text_completion(
    #     request=kwargs.get("request"),
    #     prompt=prompt,
    #     system_message=system_message,
    # )

    # Collect the complete response
    # content = [item async for item in response]
    # data = read_event_data(content)

    # Extract the text response - should already be valid JSON due to schema constraint
    # raw_json = data.raw
    raw_dict = dict(
        type=type,
        title=title,
        description=description,
        suggestion=suggestion,
        confidence=confidence,
    )

    if not raw_dict:
        raise Exception("No enhancement response generated from LLM.")

    # Return the JSON text directly - constrain_json_output ensures it's valid
    return json.dumps(raw_dict)
