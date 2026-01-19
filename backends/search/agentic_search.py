"""
AgenticSearchAgent - True agentic file search where LLM decides which tools to use.

Unlike the orchestrated SearchAgent, this implementation lets the LLM:
1. See all available tools and their descriptions
2. Decide which tool to call at each step
3. Receive results and decide the next action
4. Loop until it has enough information to answer
"""
import json
import re
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from core import common
from core.classes import FastAPIApp


# Tool definitions with descriptions for the LLM
# Note: file operations use file_index (from scan results) instead of full paths
# Examples use flat structure: {"tool": "file_read", "file_index": 0} instead of nested args
FILE_TOOLS = {
    "file_scan": {
        "description": "Scan a directory and return a numbered list of files. Use this FIRST to see available files. Each file gets an index number you can use with other tools.",
        "params": {
            "file_patterns": "Optional list of file extensions to filter, e.g. ['.pdf', '.docx']",
        },
        "returns": "Numbered list of files with metadata",
        "example": {"tool": "file_scan"},
    },
    "file_preview": {
        "description": "Get a quick preview of a file's content. Use the file_index from scan results.",
        "params": {
            "file_index": "The index number of the file from scan results (required)",
        },
        "returns": "File metadata and content preview",
        "example": {"tool": "file_preview", "file_index": 0},
    },
    "file_parse": {
        "description": "Extract full text from documents (PDF, DOCX, PPTX, XLSX). Use for binary documents.",
        "params": {
            "file_index": "The index number of the file from scan results (required)",
        },
        "returns": "Extracted document content",
        "example": {"tool": "file_parse", "file_index": 2},
    },
    "file_read": {
        "description": "Read the raw contents of a text file (.txt, .md, .json, etc.).",
        "params": {
            "file_index": "The index number of the file from scan results (required)",
        },
        "returns": "File content",
        "example": {"tool": "file_read", "file_index": 1},
    },
    "file_grep": {
        "description": "Search for a pattern within all files in the directory.",
        "params": {
            "pattern": "The search pattern (required)",
            "case_sensitive": "Case sensitive search (default: false)",
        },
        "returns": "List of matches with file index, line number, and context",
        "example": {"tool": "file_grep", "pattern": "delivery"},
    },
    "file_glob": {
        "description": "Find files matching a glob pattern.",
        "params": {
            "pattern": "Glob pattern, e.g. '*.pdf', '**/*.txt' (required)",
        },
        "returns": "List of matching files with their indices",
        "example": {"tool": "file_glob", "pattern": "*.md"},
    },
    "done": {
        "description": "Signal that you have gathered enough information. Use when ready to answer.",
        "params": {
            "summary": "Brief summary of what you found (required)",
        },
        "returns": "Ends the tool loop",
        "example": {"tool": "done", "summary": "Found 3 relevant inventory reports"},
    },
}


def _to_forward_slashes(path: str) -> str:
    """Convert Windows backslashes to forward slashes (works on all OS)."""
    return path.replace("\\", "/")

# JSON Schema for constrained tool call output - flat structure
# Example: {"tool": "file_read", "file_index": 0} instead of nested {"tool": "...", "args": {...}}
TOOL_CALL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {
            "type": "string",
            "enum": list(FILE_TOOLS.keys()),
            "description": "The tool to execute",
        },
        # Common parameters - these go directly in the object, not nested
        "file_index": {"type": "integer", "description": "File index from scan results"},
        "file_patterns": {"type": "array", "items": {"type": "string"}},
        "pattern": {"type": "string", "description": "Search or glob pattern"},
        "case_sensitive": {"type": "boolean"},
        "summary": {"type": "string", "description": "Summary for done tool"},
    },
    "required": ["tool"],
    "additionalProperties": True,  # Allow other params
}


def _build_tools_prompt() -> str:
    """Build a prompt describing all available tools with examples."""
    lines = ["Available tools:\n"]
    for tool_name, tool_info in FILE_TOOLS.items():
        lines.append(f"## {tool_name}")
        lines.append(f"{tool_info['description']}")
        if tool_info["params"]:
            lines.append("Parameters:")
            for param, desc in tool_info["params"].items():
                lines.append(f"  - {param}: {desc}")
        if "example" in tool_info:
            lines.append(f"Example: {json.dumps(tool_info['example'])}")
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
        constrain_json: dict = None,
    ) -> str:
        """
        Get a completion from the LLM.

        Args:
            prompt: The prompt to send
            system_message: Optional system message
            constrain_json: Optional JSON schema to constrain output format

        Returns:
            LLM response text (guaranteed valid JSON if constrain_json provided)
        """
        if not self.llm:
            raise ValueError("No LLM is loaded. Load a model first.")

        response = await self.llm.text_completion(
            prompt=prompt,
            system_message=system_message or "You are a helpful assistant.",
            stream=False,
            request=None,
            constrain_json_output=constrain_json,
        )

        content = []
        async for item in response:
            content.append(item)

        from inference.helpers import read_event_data
        data = read_event_data(content)
        return data.get("text", "")

    def _fix_windows_paths_in_json(self, json_str: str) -> str:
        """
        Fix Windows paths in JSON by escaping ALL backslashes within path strings.

        LLMs often return paths like C:\files\report.txt which breaks JSON parsing because:
        - \f is a valid JSON escape (form feed) → corrupts 'files' to form_feed + 'iles'
        - \r is a valid JSON escape (carriage return) → corrupts 'reports' to CR + 'eports'
        - \t is a valid JSON escape (tab) → corrupts 'temp' to tab + 'emp'
        - \n is a valid JSON escape (newline) → corrupts 'news' to newline + 'ews'
        - \b is a valid JSON escape (backspace) → corrupts 'bin' to backspace + 'in'

        Solution: Find Windows-style paths and escape ALL their backslashes.
        """

        def escape_path_backslashes(match):
            """Escape all backslashes within a matched path string."""
            path = match.group(0)
            # Replace single backslashes with double backslashes
            # But don't double-escape already-escaped backslashes
            result = ""
            i = 0
            while i < len(path):
                if path[i] == '\\':
                    # Check if already escaped (next char is also backslash)
                    if i + 1 < len(path) and path[i + 1] == '\\':
                        result += '\\\\'
                        i += 2
                    else:
                        result += '\\\\'
                        i += 1
                else:
                    result += path[i]
                    i += 1
            return result

        # Find Windows paths within JSON string values (between quotes)
        # Pattern: drive letter followed by colon and backslash, then path characters
        # This captures paths like: C:\Users\... or D:\Project Files\...
        fixed = re.sub(
            r'[A-Za-z]:\\[^"]*',
            escape_path_backslashes,
            json_str
        )

        return fixed

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
            json_str = json_match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try fixing Windows paths
                try:
                    return json.loads(self._fix_windows_paths_in_json(json_str))
                except json.JSONDecodeError:
                    pass

        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Try fixing Windows paths
                try:
                    return json.loads(self._fix_windows_paths_in_json(json_str))
                except json.JSONDecodeError:
                    pass

        return None

    async def search(
        self,
        query: str,
        directory: str,
        max_iterations: int = 10,
        file_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute an agentic search where the LLM decides which tools to use.

        Args:
            query: The user's search query
            directory: The starting directory to search in
            max_iterations: Maximum number of tool calls before stopping
            file_patterns: Optional hint about file types to focus on

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
        # File index map: index (int) -> file info dict with 'path', 'filename', etc.
        file_index_map: Dict[int, Dict[str, Any]] = {}

        # Build initial context - don't show full directory path, just indicate scanning is available
        initial_context = f"""You are a file search agent. Your task is to find information to answer the user's query.

User Query: {query}

{f"File type hints: {file_patterns}" if file_patterns else ""}

{TOOLS_PROMPT}

## Instructions
1. Start by using file_scan to see available files (each file gets an index number)
2. Use file_index to reference files in other tools (e.g., file_read, file_preview)
3. When you have enough information, use the "done" tool
4. Always respond with a JSON object

What tool would you like to use first?"""

        system_message = "You are a file search agent. Respond with JSON. Use file_index (integer) to reference files from scan results. Start with file_scan, then read relevant files."

        current_prompt = initial_context

        for iteration in range(max_iterations):
            print(f"{common.PRNT_API} [AgenticSearch] Iteration {iteration + 1}/{max_iterations}", flush=True)

            # Get LLM's tool choice with constrained JSON output
            try:
                llm_response = await self._llm_completion(
                    prompt=current_prompt,
                    system_message=system_message,
                    constrain_json=TOOL_CALL_SCHEMA,
                )
            except Exception as e:
                tool_logs.append({
                    "iteration": iteration + 1,
                    "error": f"LLM error: {e}",
                })
                break

            # Parse tool call - with constrained output this should always succeed
            try:
                tool_call = json.loads(llm_response)
            except json.JSONDecodeError:
                # Fallback to regex parsing if somehow constrained output failed
                tool_call = self._parse_tool_call(llm_response)

            if not tool_call:
                tool_logs.append({
                    "iteration": iteration + 1,
                    "error": "Failed to parse tool call",
                    "raw_response": llm_response[:500],
                })
                # Retry without constrained output as fallback
                current_prompt = f"Select a tool from: {list(FILE_TOOLS.keys())}. Respond with JSON."
                continue

            tool_name = tool_call.get("tool")
            # Flat format: {"tool": "file_read", "file_index": 0}
            tool_args = {k: v for k, v in tool_call.items() if k != "tool"}
            # If LLM still uses nested args, flatten it
            if "args" in tool_args and isinstance(tool_args["args"], dict):
                tool_args = tool_args["args"]

            # Normalize common tool name variations
            TOOL_ALIASES = {
                "scan": "file_scan",
                "preview": "file_preview",
                "parse": "file_parse",
                "read": "file_read",
                "read_file": "file_read",
                "grep": "file_grep",
                "glob": "file_glob",
                "search": "file_grep",
            }
            if tool_name in TOOL_ALIASES:
                tool_name = TOOL_ALIASES[tool_name]

            # Normalize argument names (LLM may use variations)
            ARG_ALIASES = {
                "index": "file_index",
                "idx": "file_index",
                "file": "file_index",
                "filename": "file_index",
            }
            for alias, canonical in ARG_ALIASES.items():
                if alias in tool_args and canonical not in tool_args:
                    tool_args[canonical] = tool_args.pop(alias)

            # Resolve file_index to file_path for file operations
            if tool_name in ["file_preview", "file_read", "file_parse"]:
                file_index = tool_args.get("file_index")
                if file_index is not None:
                    resolved_path = None

                    # Try to parse as integer first
                    try:
                        idx = int(file_index)
                        if idx in file_index_map:
                            resolved_path = file_index_map[idx]["path"]
                    except (ValueError, TypeError):
                        pass

                    # If not an integer or not found, try to match by filename
                    if resolved_path is None and isinstance(file_index, str) and file_index_map:
                        # Try exact match on filename or relative_path
                        for idx, info in file_index_map.items():
                            filename = info.get("filename", "")
                            rel_path = info.get("relative_path", "")
                            if file_index == filename or file_index == rel_path:
                                resolved_path = info["path"]
                                print(f"{common.PRNT_API} [AgenticSearch] Matched filename '{file_index}' to index {idx}", flush=True)
                                break
                        # Try partial match if exact match failed
                        if resolved_path is None:
                            for idx, info in file_index_map.items():
                                filename = info.get("filename", "")
                                if file_index in filename or filename in file_index:
                                    resolved_path = info["path"]
                                    print(f"{common.PRNT_API} [AgenticSearch] Partial match '{file_index}' to '{filename}' (index {idx})", flush=True)
                                    break

                    if resolved_path:
                        tool_args["file_path"] = resolved_path
                        del tool_args["file_index"]
                    else:
                        available = ", ".join(str(i) for i in sorted(file_index_map.keys())[:10])
                        raise ValueError(f"Could not resolve file_index '{file_index}'. Use integer indices: {available}")

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
                # Inject default directory for tools that need it
                if tool_name in ["file_scan", "file_grep", "file_glob"]:
                    if "directory_path" not in tool_args or not tool_args["directory_path"]:
                        tool_args["directory_path"] = directory

                # Check for missing file_path on file operations (after index resolution)
                if tool_name in ["file_preview", "file_read", "file_parse"]:
                    if "file_path" not in tool_args or not tool_args["file_path"]:
                        if file_index_map:
                            available = ", ".join(str(i) for i in sorted(file_index_map.keys())[:10])
                            raise ValueError(
                                f"file_index is required. Use an index from scan results: {available}"
                            )
                        else:
                            raise ValueError(
                                "Use file_scan first to see available files, then use file_index to reference them."
                            )

                result = await self._execute_tool(tool_name, **tool_args)

                # Build file index map from scan results
                if tool_name == "file_scan" and isinstance(result, list):
                    file_index_map.clear()
                    for idx, file_info in enumerate(result):
                        file_index_map[idx] = file_info
                    # Format scan results for LLM with indices and forward slashes (no full paths!)
                    formatted_files = []
                    for idx, f in enumerate(result):
                        filename = f.get("filename", f.get("relative_path", "unknown"))
                        file_type = f.get("type", "file")
                        size = f.get("size", "")
                        formatted_files.append(f"[{idx}] {filename} ({file_type}, {size})")
                    result_str = "Files found:\n" + "\n".join(formatted_files)
                else:
                    # Track sources from file operations
                    if tool_name in ["file_read", "file_parse"]:
                        file_path = tool_args.get("file_path", "")
                        if file_path:
                            sources.add(file_path)

                    # Standard JSON result for non-scan tools
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

                # For context history shown to LLM, use simplified args (no full paths)
                display_args = {k: v for k, v in tool_args.items() if k != "directory_path"}
                if "file_path" in display_args:
                    # Show filename instead of full path
                    display_args["file"] = Path(display_args["file_path"]).name
                    del display_args["file_path"]

                context_history.append({
                    "tool": tool_name,
                    "args": display_args,
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

What tool would you like to use next? Use file_index (integer) to reference files.
If you have enough information, use the "done" tool."""

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

        return {
            "answer": answer,
            "sources": list(sources),
            "tool_logs": tool_logs,
            "iterations": len([l for l in tool_logs if "tool" in l]),
        }
