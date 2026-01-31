from typing import List, Literal
from pydantic import BaseModel, Field


class Params(BaseModel):
    """Search across multiple data sources. Use this tool to find information in memos, project files, system files, web, or conversation history."""

    query: str = Field(
        ...,
        description="The search query or question to find answers for.",
    )
    sources: List[Literal["memos", "project_files", "sys_files", "web", "convos"]] = Field(
        default=["memos", "project_files"],
        description="Data sources to search: 'memos' for memory collections, 'project_files' for project documents, 'sys_files' for filesystem, 'web' for internet search, 'convos' for conversation history.",
        options=["memos", "project_files", "sys_files", "web", "convos"],
    )
    max_results: int = Field(
        default=10,
        description="Maximum number of results to return per source.",
    )
    search_type: Literal["semantic", "keyword", "hybrid"] = Field(
        default="semantic",
        description="Type of search: 'semantic' for meaning-based, 'keyword' for exact matches, 'hybrid' for both.",
        options=["semantic", "keyword", "hybrid"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "How does the authentication system work?",
                    "sources": ["project_files", "memos"],
                    "max_results": 5,
                    "search_type": "semantic",
                }
            ]
        }
    }


async def main(**kwargs: Params) -> dict:
    query = kwargs.get("query")
    sources = kwargs.get("sources", ["memos", "project_files"])
    max_results = kwargs.get("max_results", 10)
    search_type = kwargs.get("search_type", "semantic")

    if not query:
        raise ValueError("A search query is required.")

    if not sources:
        raise ValueError("At least one search source must be specified.")

    # Validate sources
    valid_sources = {"memos", "project_files", "sys_files", "web", "convos"}
    for source in sources:
        if source not in valid_sources:
            raise ValueError(f"Invalid source: {source}. Valid options are: {valid_sources}")

    return {
        "query": query,
        "sources": sources,
        "max_results": max_results,
        "search_type": search_type,
        "status": "ready",
    }
