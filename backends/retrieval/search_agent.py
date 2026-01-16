"""
SearchAgent - Agentic file search with multi-phase strategy.

This agent orchestrates file search tools to explore directories and find
relevant information using a multi-phase approach:
1. SCAN: Get an overview of available files
2. PREVIEW: Quick look at potentially relevant files
3. DEEP DIVE: Full parse of the most relevant files
4. SYNTHESIZE: LLM generates final answer from collected context
"""
import hashlib
import json
import re
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from core import common
from core.classes import FastAPIApp


# JSON Schema for constrained file selection output - just an array of indices
FILE_SELECTION_SCHEMA = {
    "type": "array",
    "items": {"type": "integer"},
    "description": "Array of file indices to select",
}


class SearchAgent:
    """
    Agentic file search with multi-phase strategy and directory whitelisting.

    The agent uses a combination of file tools to explore directories and find
    relevant information, then synthesizes an answer using the LLM.
    """

    def __init__(
        self,
        app: FastAPIApp,
        allowed_directories: List[str],
    ):
        """
        Initialize the SearchAgent.

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
        """
        Ensure a path is within the allowed directories.

        Args:
            path: Path to validate

        Returns:
            True if path is allowed, False otherwise
        """
        try:
            resolved = Path(path).resolve()
            return any(
                resolved == allowed or resolved.is_relative_to(allowed)
                for allowed in self.allowed_directories
            )
        except (ValueError, OSError):
            return False

    def _load_tool(self, tool_name: str):
        """
        Dynamically load a file tool module.

        Args:
            tool_name: Name of the tool to load

        Returns:
            The tool module
        """
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
        """
        Execute a file tool with the given arguments.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Arguments to pass to the tool

        Returns:
            Tool execution result
        """
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
            prompt: The prompt to send to the LLM
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

        # Collect response
        content = []
        async for item in response:
            content.append(item)

        # Parse response
        from inference.helpers import read_event_data
        data = read_event_data(content)
        return data.get("text", "")

    async def search(
        self,
        query: str,
        directory: str,
        max_files_preview: int = 10,
        max_files_parse: int = 3,
        file_patterns: Optional[List[str]] = None,
        cache_results: bool = False,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a multi-phase agentic search.

        Args:
            query: The user's search query
            directory: The directory to search in
            max_files_preview: Maximum number of files to preview
            max_files_parse: Maximum number of files to fully parse
            file_patterns: Optional list of file extensions to filter
            cache_results: Whether to cache parsed documents in ChromaDB
            collection_name: Collection name for caching (required if cache_results=True)

        Returns:
            Dict with: answer, sources, tool_logs
        """
        # Validate directory
        if not self._validate_path(directory):
            raise ValueError(
                f"Access denied: '{directory}' is outside allowed directories"
            )

        tool_logs = []
        collected_context = []
        sources = []

        # Phase 1: SCAN - Get directory overview
        print(f"{common.PRNT_API} [SearchAgent] Phase 1: Scanning directory", flush=True)
        try:
            scan_result = await self._execute_tool(
                "file_scan",
                directory_path=directory,
                file_patterns=file_patterns,
                recursive=True,
            )
            tool_logs.append({
                "phase": "scan",
                "tool": "file_scan",
                "result_count": len(scan_result),
            })
        except Exception as e:
            return {
                "answer": f"Failed to scan directory: {e}",
                "sources": [],
                "tool_logs": tool_logs,
                "error": str(e),
            }

        if not scan_result:
            return {
                "answer": "No files found in the specified directory.",
                "sources": [],
                "tool_logs": tool_logs,
            }

        # Build file index map (index -> file info) - LLM sees indices, not paths
        file_index_map: Dict[int, Dict[str, Any]] = {}
        for idx, file_info in enumerate(scan_result[:50]):  # Limit to first 50
            file_index_map[idx] = file_info

        # Phase 2: Use LLM to select files to preview based on query
        print(f"{common.PRNT_API} [SearchAgent] Phase 2: Selecting files to preview", flush=True)

        # Format files with indices for LLM (no full paths exposed)
        file_list_str = "\n".join(
            f"[{idx}] {f.get('filename', f.get('relative_path', 'unknown'))} ({f.get('type', 'file')}, {f.get('size', '')})"
            for idx, f in file_index_map.items()
        )

        selection_prompt = f"""Given the user's query and a list of available files, select the most relevant files to examine.

User Query: {query}

Available Files:
{file_list_str}

Instructions:
1. Select up to {max_files_preview} files that are most likely to contain information relevant to the query
2. Prioritize files by relevance to the query
3. Return ONLY a JSON array of file indices (the numbers in brackets)
4. Example: [0, 2, 5]"""

        try:
            selection_response = await self._llm_completion(
                prompt=selection_prompt,
                system_message="You are a file selection assistant. Return only a JSON array of integers.",
                constrain_json=FILE_SELECTION_SCHEMA,
            )

            # Parse the selection - with constrained output this should always succeed
            try:
                selected_indices = json.loads(selection_response)
                # Handle if LLM still wraps in object
                if isinstance(selected_indices, dict):
                    selected_indices = selected_indices.get("files", [])
            except json.JSONDecodeError:
                # Fallback to regex parsing
                json_match = re.search(r'\[.*\]', selection_response, re.DOTALL)
                if json_match:
                    selected_indices = json.loads(json_match.group())
                else:
                    selected_indices = list(range(min(max_files_preview, len(file_index_map))))

            # Validate indices are within range
            selected_indices = [
                idx for idx in selected_indices
                if isinstance(idx, int) and idx in file_index_map
            ]

            tool_logs.append({
                "phase": "select",
                "selected_count": len(selected_indices),
                "selected_indices": selected_indices,
            })
        except Exception as e:
            print(f"{common.PRNT_API} [SearchAgent] Selection failed, using fallback: {e}", flush=True)
            selected_indices = list(range(min(max_files_preview, len(file_index_map))))

        # Phase 3: PREVIEW - Quick look at selected files
        print(f"{common.PRNT_API} [SearchAgent] Phase 3: Previewing {len(selected_indices)} files", flush=True)
        previews = []
        preview_index_map: Dict[int, Dict[str, Any]] = {}  # preview_idx -> preview info

        for file_idx in selected_indices[:max_files_preview]:
            file_info = file_index_map.get(file_idx)
            if not file_info:
                continue
            full_path = file_info.get("path")
            rel_path = file_info.get("relative_path", file_info.get("filename", "unknown"))
            try:
                preview = await self._execute_tool(
                    "file_preview",
                    file_path=full_path,
                    max_chars=500,
                    max_lines=20,
                )
                preview_data = {
                    "file_index": file_idx,
                    "path": full_path,
                    "relative_path": rel_path,
                    "filename": file_info.get("filename", rel_path),
                    "preview": preview,
                }
                previews.append(preview_data)
                preview_index_map[len(previews) - 1] = preview_data
            except Exception as e:
                print(f"{common.PRNT_API} [SearchAgent] Preview failed for [{file_idx}]: {e}", flush=True)

        tool_logs.append({
            "phase": "preview",
            "tool": "file_preview",
            "previewed_count": len(previews),
        })

        # Phase 4: Use LLM to select files for deep parsing
        print(f"{common.PRNT_API} [SearchAgent] Phase 4: Selecting files for deep parse", flush=True)

        # Format previews with indices for LLM (using preview list index)
        preview_summaries = "\n\n".join(
            f"[{idx}] {p['filename']}\nRequires Parsing: {p['preview'].get('requires_parsing', False)}\nPreview:\n{p['preview'].get('preview', '[No preview]')[:300]}"
            for idx, p in enumerate(previews)
        )

        deep_parse_prompt = f"""Based on the file previews, select the most relevant files for full content extraction.

User Query: {query}

File Previews:
{preview_summaries}

Instructions:
1. Select up to {max_files_parse} files that are most likely to answer the user's query
2. Return ONLY a JSON array of file indices (the numbers in brackets)
3. Example: [0, 1]"""

        try:
            parse_selection = await self._llm_completion(
                prompt=deep_parse_prompt,
                system_message="You are a file selection assistant. Return only a JSON array of integers.",
                constrain_json=FILE_SELECTION_SCHEMA,
            )

            # Parse - with constrained output this should always succeed
            try:
                parse_indices = json.loads(parse_selection)
                # Handle if LLM still wraps in object
                if isinstance(parse_indices, dict):
                    parse_indices = parse_indices.get("files", [])
            except json.JSONDecodeError:
                # Fallback to regex parsing
                json_match = re.search(r'\[.*\]', parse_selection, re.DOTALL)
                if json_match:
                    parse_indices = json.loads(json_match.group())
                else:
                    parse_indices = list(range(min(max_files_parse, len(previews))))

            # Validate indices are within preview list range
            parse_indices = [
                idx for idx in parse_indices
                if isinstance(idx, int) and 0 <= idx < len(previews)
            ]

            tool_logs.append({
                "phase": "deep_parse_select",
                "selected_count": len(parse_indices),
                "selected_indices": parse_indices,
            })
        except Exception:
            parse_indices = list(range(min(max_files_parse, len(previews))))

        # Phase 5: DEEP DIVE - Full parse of selected files
        print(f"{common.PRNT_API} [SearchAgent] Phase 5: Deep parsing {len(parse_indices)} files", flush=True)
        for preview_idx in parse_indices[:max_files_parse]:
            if preview_idx >= len(previews):
                continue

            preview_data = previews[preview_idx]
            full_path = preview_data["path"]
            rel_path = preview_data["relative_path"]
            filename = preview_data["filename"]

            # Decide whether to parse or read based on preview
            requires_parsing = preview_data["preview"].get("requires_parsing", False)

            try:
                if requires_parsing:
                    # Use file_parse for documents
                    content = await self._execute_tool(
                        "file_parse",
                        file_path=full_path,
                        output_format="text",
                    )
                    extracted_text = content.get("content", "")
                    tool_used = "file_parse"
                else:
                    # Use file_read for text files
                    content = await self._execute_tool(
                        "file_read",
                        file_path=full_path,
                    )
                    extracted_text = content.get("content", "")
                    tool_used = "file_read"

                collected_context.append({
                    "source": filename,
                    "content": extracted_text[:5000],  # Limit context size
                })
                sources.append(rel_path)

                tool_logs.append({
                    "phase": "parse",
                    "tool": tool_used,
                    "file": filename,
                    "content_length": len(extracted_text),
                })
            except Exception as e:
                print(f"{common.PRNT_API} [SearchAgent] Parse failed for [{preview_idx}] {filename}: {e}", flush=True)
                tool_logs.append({
                    "phase": "parse",
                    "file": filename,
                    "error": str(e),
                })

        # Phase 6: SYNTHESIZE - Generate final answer
        print(f"{common.PRNT_API} [SearchAgent] Phase 6: Synthesizing answer", flush=True)
        if not collected_context:
            return {
                "answer": "Could not extract relevant content from the files. Please try a different query or directory.",
                "sources": sources,
                "tool_logs": tool_logs,
            }

        context_str = "\n\n---\n\n".join(
            f"Source: {c['source']}\n\n{c['content']}"
            for c in collected_context
        )

        synthesis_prompt = f"""Based on the following document contents, answer the user's query.

User Query: {query}

Document Contents:
{context_str}

Instructions:
1. Answer the query based ONLY on the information in the documents above
2. If the documents don't contain relevant information, say so
3. Cite specific sources when making claims
4. Be concise but thorough"""

        try:
            answer = await self._llm_completion(
                prompt=synthesis_prompt,
                system_message="You are a helpful assistant that answers questions based on provided document contents. Always cite your sources.",
            )
        except Exception as e:
            answer = f"Failed to synthesize answer: {e}"

        # Optional: Cache results in ChromaDB
        if cache_results and collection_name:
            try:
                await self._cache_documents(
                    documents=collected_context,
                    collection_name=collection_name,
                )
                tool_logs.append({
                    "phase": "cache",
                    "collection": collection_name,
                    "documents_cached": len(collected_context),
                })
            except Exception as e:
                print(f"{common.PRNT_API} [SearchAgent] Caching failed: {e}", flush=True)

        return {
            "answer": answer,
            "sources": sources,
            "tool_logs": tool_logs,
            "files_scanned": len(scan_result),
            "files_previewed": len(previews),
            "files_parsed": len(collected_context),
        }

    async def _cache_documents(
        self,
        documents: List[Dict[str, str]],
        collection_name: str,
    ):
        """
        Cache parsed documents in ChromaDB for future RAG queries.

        Args:
            documents: List of {"source": path, "content": text}
            collection_name: Target collection name
        """
        from embeddings.vector_storage import Vector_Storage
        from embeddings.embedder import Embedder

        vector_storage = Vector_Storage(app=self.app)
        embedder = Embedder(app=self.app)

        # Get or create collection
        try:
            collection = vector_storage.db_client.get_collection(name=collection_name)
        except Exception:
            collection = vector_storage.db_client.create_collection(
                name=collection_name,
                metadata={"type": "search_agent_cache"},
            )

        # Add documents
        for doc in documents:
            # embed_text is synchronous
            embedding = embedder.embed_text(doc["content"][:2000])  # Limit for embedding
            # Use deterministic hash for document ID
            doc_id = hashlib.sha256(doc["source"].encode()).hexdigest()[:16]
            collection.add(
                embeddings=[embedding],
                documents=[doc["content"]],
                metadatas=[{"source": doc["source"]}],
                ids=[f"search_{doc_id}"],
            )
