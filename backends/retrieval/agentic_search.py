"""
AgenticSearchAgent - True agentic file search where LLM decides which tools to use.

Unlike the orchestrated SearchAgent, this implementation lets the LLM:
1. See all available tools and their descriptions
2. Decide which tool to call at each step
3. Receive results and decide the next action
4. Loop until it has enough information to answer
"""
import hashlib
import json
import re
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from core import common
from core.classes import FastAPIApp


# Tool definitions with descriptions for the LLM
FILE_TOOLS = {
    "file_scan": {
        "description": "Scan a directory and return metadata about all files (name, type, size, modified date). Use this first to get an overview of available files.",
        "params": {
            "directory_path": "The directory path to scan (required)",
            "file_patterns": "Optional list of file extensions to filter, e.g. ['.pdf', '.docx']",
            "recursive": "Whether to scan subdirectories (default: True)",
        },
        "returns": "List of file metadata objects",
    },
    "file_preview": {
        "description": "Get a quick preview of a file's content (first ~500 chars). Use this to quickly assess if a file is relevant before fully parsing it.",
        "params": {
            "file_path": "The path to the file to preview (required)",
            "max_chars": "Maximum characters to preview (default: 500)",
            "max_lines": "Maximum lines to preview (default: 20)",
        },
        "returns": "File metadata and content preview",
    },
    "file_parse": {
        "description": "Extract full text from documents (PDF, DOCX, PPTX, XLSX, RTF, CSV, XML, HTML). Use this for binary documents that need parsing.",
        "params": {
            "file_path": "The path to the document (required)",
            "output_format": "'markdown' or 'text' (default: 'text')",
        },
        "returns": "Extracted document content",
    },
    "file_read": {
        "description": "Read the raw contents of a text file. Use this for plain text files (.txt, .md, .py, etc.).",
        "params": {
            "file_path": "The path to the file (required)",
            "start_line": "Starting line number, 1-indexed (optional)",
            "end_line": "Ending line number, inclusive (optional)",
        },
        "returns": "File content and line info",
    },
    "file_grep": {
        "description": "Search for a pattern (text or regex) within files in a directory. Use this to find specific content across multiple files.",
        "params": {
            "pattern": "The search pattern (required)",
            "directory_path": "The directory to search (required)",
            "file_patterns": "Optional file extensions to search",
            "case_sensitive": "Case sensitive search (default: False)",
            "use_regex": "Interpret pattern as regex (default: False)",
            "max_results": "Maximum matches to return (default: 50)",
        },
        "returns": "List of matches with file, line number, and context",
    },
    "file_glob": {
        "description": "Find files matching a glob pattern. Use this to locate files by name pattern.",
        "params": {
            "pattern": "Glob pattern, e.g. '*.pdf', '**/*.txt' (required)",
            "directory_path": "The directory to search (required)",
        },
        "returns": "List of matching file paths",
    },
    "done": {
        "description": "Signal that you have gathered enough information to answer the query. Use this when you're ready to provide the final answer.",
        "params": {
            "summary": "Brief summary of what you found (required)",
        },
        "returns": "Ends the tool loop",
    },
}


def _build_tools_prompt() -> str:
    """Build a prompt describing all available tools."""
    lines = ["Available tools:\n"]
    for tool_name, tool_info in FILE_TOOLS.items():
        lines.append(f"## {tool_name}")
        lines.append(f"Description: {tool_info['description']}")
        lines.append("Parameters:")
        for param, desc in tool_info["params"].items():
            lines.append(f"  - {param}: {desc}")
        lines.append(f"Returns: {tool_info['returns']}")
        lines.append("")
    return "\n".join(lines)


TOOLS_PROMPT = _build_tools_prompt()


class AgenticSearchAgent:
    """
    True agentic file search where the LLM decides which tools to use.

    The agent presents tools to the LLM and lets it decide:
    - Which tool to call next
    - What arguments to pass
    - When it has enough information (calls "done")
    """

    def __init__(
        self,
        app: FastAPIApp,
        allowed_directories: List[str],
    ):
        """
        Initialize the AgenticSearchAgent.

        Args:
            app: FastAPI application instance with LLM access
            allowed_directories: List of directory paths the agent is allowed to access
        """
        self.app = app
        self.llm = app.state.llm
        self.allowed_directories = [
            Path(d).resolve() for d in allowed_directories
        ]
        self._tool_cache: Dict[str, Any] = {}

    def _validate_path(self, path: str) -> bool:
        """Ensure a path is within the allowed directories."""
        try:
            resolved = Path(path).resolve()
            return any(
                resolved == allowed or resolved.is_relative_to(allowed)
                for allowed in self.allowed_directories
            )
        except (ValueError, OSError):
            return False

    def _load_tool(self, tool_name: str):
        """Dynamically load a file tool module."""
        if tool_name in self._tool_cache:
            return self._tool_cache[tool_name]

        try:
            module = importlib.import_module(
                f"tools.built_in_functions.{tool_name}"
            )
            self._tool_cache[tool_name] = module
            return module
        except ImportError as e:
            raise ValueError(f"Failed to load tool '{tool_name}': {e}")

    async def _execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a file tool with the given arguments."""
        # Validate paths in kwargs
        for key in ["directory_path", "file_path"]:
            if key in kwargs and kwargs[key]:
                if not self._validate_path(kwargs[key]):
                    raise ValueError(
                        f"Access denied: '{kwargs[key]}' is outside allowed directories"
                    )

        tool = self._load_tool(tool_name)
        return await tool.main(**kwargs)

    async def _llm_completion(
        self,
        prompt: str,
        system_message: str = None,
    ) -> str:
        """Get a completion from the LLM."""
        if not self.llm:
            raise ValueError("No LLM is loaded. Load a model first.")

        response = await self.llm.text_completion(
            prompt=prompt,
            system_message=system_message or "You are a helpful assistant.",
            stream=False,
            request=None,
        )

        content = []
        async for item in response:
            content.append(item)

        from inference.helpers import read_event_data
        data = read_event_data(content)
        return data.get("text", "")

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM's response to extract tool call.

        Expected format:
        ```json
        {"tool": "tool_name", "args": {"param1": "value1", ...}}
        ```
        """
        # Try to find JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return None

    async def search(
        self,
        query: str,
        directory: str,
        max_iterations: int = 10,
        file_patterns: Optional[List[str]] = None,
        cache_results: bool = False,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agentic search where the LLM decides which tools to use.

        Args:
            query: The user's search query
            directory: The starting directory to search in
            max_iterations: Maximum number of tool calls before stopping
            file_patterns: Optional hint about file types to focus on
            cache_results: Whether to cache parsed documents in ChromaDB
            collection_name: Collection name for caching

        Returns:
            Dict with: answer, sources, tool_logs, iterations
        """
        if not self._validate_path(directory):
            raise ValueError(
                f"Access denied: '{directory}' is outside allowed directories"
            )

        tool_logs = []
        context_history = []
        sources = set()

        # Build initial context
        initial_context = f"""You are a file search agent. Your task is to find information to answer the user's query.

User Query: {query}

Starting Directory: {directory}
{f"File type hints: {file_patterns}" if file_patterns else ""}

{TOOLS_PROMPT}

## Instructions
1. Analyze the query and decide which tool to use first
2. After each tool result, decide if you need more information or can answer
3. When you have enough information, use the "done" tool
4. Always respond with a JSON tool call in this format:

```json
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}
```

What tool would you like to use first?"""

        system_message = "You are a file search agent. Always respond with a JSON tool call. Be methodical - scan first, then preview relevant files, then read/parse as needed."

        current_prompt = initial_context

        for iteration in range(max_iterations):
            print(f"{common.PRNT_API} [AgenticSearch] Iteration {iteration + 1}/{max_iterations}", flush=True)

            # Get LLM's tool choice
            try:
                llm_response = await self._llm_completion(
                    prompt=current_prompt,
                    system_message=system_message,
                )
            except Exception as e:
                tool_logs.append({
                    "iteration": iteration + 1,
                    "error": f"LLM error: {e}",
                })
                break

            # Parse tool call
            tool_call = self._parse_tool_call(llm_response)

            if not tool_call:
                # LLM didn't return valid tool call - try to recover
                tool_logs.append({
                    "iteration": iteration + 1,
                    "error": "Failed to parse tool call",
                    "raw_response": llm_response[:500],
                })
                # Prompt LLM to try again
                current_prompt = f"""Your previous response was not a valid JSON tool call. Please respond with:

```json
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}
```

Available tools: {list(FILE_TOOLS.keys())}

What tool would you like to use?"""
                continue

            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("args", {})

            print(f"{common.PRNT_API} [AgenticSearch] Tool: {tool_name}, Args: {tool_args}", flush=True)

            # Check if done
            if tool_name == "done":
                tool_logs.append({
                    "iteration": iteration + 1,
                    "tool": "done",
                    "summary": tool_args.get("summary", ""),
                })
                break

            # Validate tool name
            if tool_name not in FILE_TOOLS or tool_name == "done":
                tool_logs.append({
                    "iteration": iteration + 1,
                    "error": f"Unknown tool: {tool_name}",
                })
                current_prompt = f"Unknown tool '{tool_name}'. Available tools: {list(FILE_TOOLS.keys())}. Please choose a valid tool."
                continue

            # Execute tool
            try:
                result = await self._execute_tool(tool_name, **tool_args)

                # Track sources from file operations
                if tool_name in ["file_read", "file_parse"]:
                    file_path = tool_args.get("file_path", "")
                    if file_path:
                        sources.add(file_path)

                # Truncate large results for context
                result_str = json.dumps(result, indent=2, default=str)
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + "\n... (truncated)"

                tool_logs.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "success": True,
                    "result_preview": result_str[:500] if len(result_str) > 500 else result_str,
                })

                context_history.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_str,
                })

            except Exception as e:
                error_msg = str(e)
                tool_logs.append({
                    "iteration": iteration + 1,
                    "tool": tool_name,
                    "args": tool_args,
                    "success": False,
                    "error": error_msg,
                })
                result_str = f"Error: {error_msg}"
                context_history.append({
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result_str,
                })

            # Build next prompt with results
            history_str = "\n\n".join(
                f"Tool: {h['tool']}\nArgs: {h['args']}\nResult:\n{h['result']}"
                for h in context_history[-5:]  # Keep last 5 for context length
            )

            current_prompt = f"""Previous tool results:

{history_str}

Based on these results, what tool would you like to use next?
If you have enough information to answer the query, use the "done" tool.

Remember to respond with a JSON tool call:
```json
{{"tool": "tool_name", "args": {{"param1": "value1"}}}}
```"""

        # Final synthesis
        print(f"{common.PRNT_API} [AgenticSearch] Synthesizing final answer", flush=True)

        if not context_history:
            return {
                "answer": "Could not gather any information. Please try a different query.",
                "sources": list(sources),
                "tool_logs": tool_logs,
                "iterations": len(tool_logs),
            }

        # Build synthesis prompt from gathered context
        synthesis_context = "\n\n---\n\n".join(
            f"Tool: {h['tool']}\nResult:\n{h['result']}"
            for h in context_history
        )

        synthesis_prompt = f"""Based on the information gathered by the search agent, answer the user's query.

User Query: {query}

Information Gathered:
{synthesis_context}

Instructions:
1. Answer the query based ONLY on the information above
2. If the information doesn't fully answer the query, say what's missing
3. Cite specific files when making claims
4. Be concise but thorough"""

        try:
            answer = await self._llm_completion(
                prompt=synthesis_prompt,
                system_message="You are a helpful assistant. Answer based only on the provided information.",
            )
        except Exception as e:
            answer = f"Failed to synthesize answer: {e}"

        # Optional caching
        if cache_results and collection_name:
            try:
                await self._cache_results(context_history, collection_name)
                tool_logs.append({
                    "phase": "cache",
                    "collection": collection_name,
                })
            except Exception as e:
                print(f"{common.PRNT_API} [AgenticSearch] Caching failed: {e}", flush=True)

        return {
            "answer": answer,
            "sources": list(sources),
            "tool_logs": tool_logs,
            "iterations": len([l for l in tool_logs if "tool" in l]),
        }

    async def _cache_results(
        self,
        context_history: List[Dict],
        collection_name: str,
    ):
        """Cache gathered content in ChromaDB."""
        from embeddings.vector_storage import Vector_Storage
        from embeddings.embedder import Embedder

        vector_storage = Vector_Storage(app=self.app)
        embedder = Embedder(app=self.app)

        try:
            collection = vector_storage.db_client.get_collection(name=collection_name)
        except Exception:
            collection = vector_storage.db_client.create_collection(
                name=collection_name,
                metadata={"type": "agentic_search_cache"},
            )

        for entry in context_history:
            if entry["tool"] in ["file_read", "file_parse"]:
                content = entry.get("result", "")[:2000]
                if content and not content.startswith("Error:"):
                    embedding = embedder.embed_text(content)
                    doc_id = hashlib.sha256(content[:100].encode()).hexdigest()[:16]
                    collection.add(
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[{"tool": entry["tool"], "args": str(entry["args"])}],
                        ids=[f"agentic_{doc_id}"],
                    )
