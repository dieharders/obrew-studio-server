# Human-in-the-Loop Search Agent

## Overview

Add a feedback mechanism to the search agents so they can pause and request user input when needed. This enables workflow integration where external systems can handle user prompts.

## Design

### Response Model with Input Request

When the agent needs user input, the response includes:

```python
{
    "success": True,
    "status": "input_required",  # "complete" | "input_required" | "error"
    "data": {
        "answer": None,           # Only populated when complete
        "sources": [...],
        "tool_logs": [...],
    },
    "input_required": {
        "prompt": "I couldn't find inventory reports in this directory. Where else should I look?",
        "type": "directory",      # "directory" | "confirmation" | "choice" | "text"
        "options": [...],         # Optional for "choice" type
        "session_id": "uuid",     # For resuming
    },
    "state": {...}                # Serialized session state for resume
}
```

### Session Harness: `SearchSession`

A wrapper class that manages state and enables resume:

```python
class SearchSession:
    """Manages search agent lifecycle with human-in-the-loop support."""

    def __init__(self, app, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.state = {}  # Serializable agent state

    async def start(self, query, directory, ...) -> SearchResponse:
        """Start a new search session."""

    async def resume(self, user_input: str) -> SearchResponse:
        """Resume with user-provided input."""

    def to_dict(self) -> dict:
        """Serialize state for storage."""

    @classmethod
    def from_dict(cls, data: dict) -> "SearchSession":
        """Restore from serialized state."""
```

### State Management (Client-Side)

State is returned to the client and sent back on resume (stateless server):

```python
state = {
    "session_id": "uuid",
    "agent_type": "orchestrated",  # or "agentic"
    "query": "original query",
    "directory": "current directory",
    "allowed_directories": [...],
    "file_index_map": {...},      # Scan results
    "context_history": [...],      # Tool call history
    "sources": [...],
    "current_phase": "scan",       # For orchestrated agent
    "current_iteration": 3,        # For agentic agent
    "pending_input": {
        "type": "directory",
        "reason": "no_results"
    }
}
```

**Scope:** Both `SearchAgent` and `AgenticSearchAgent` will support this.

## Implementation Steps

### Step 1: Add Response Models

In `backends/inference/classes.py`:

```python
class InputRequired(BaseModel):
    prompt: str
    type: Literal["directory", "confirmation", "choice", "text"]
    options: Optional[List[str]] = None
    session_id: str

class SearchResponse(BaseModel):
    success: bool
    status: Literal["complete", "input_required", "error"]
    message: Optional[str] = None
    data: Optional[dict] = None
    input_required: Optional[InputRequired] = None
    state: Optional[dict] = None  # For resume
```

### Step 2: Create SearchSession Class

New file: `backends/retrieval/search_session.py`

```python
class SearchSession:
    def __init__(self, app, agent_type: str = "orchestrated"):
        self.session_id = str(uuid.uuid4())
        self.app = app
        self.agent_type = agent_type
        self.state = {
            "file_index_map": {},
            "context_history": [],
            "sources": [],
            "iteration": 0,
        }

    async def run(self, query: str, directory: str, ...) -> dict:
        """Run search, may return input_required."""

    async def resume(self, user_input: str, input_type: str) -> dict:
        """Resume with user input (e.g., new directory path)."""

    def _should_request_input(self, phase: str, result: Any) -> Optional[dict]:
        """Decide if we need user input based on current state."""
        # Examples:
        # - No files found in scan → ask for different directory
        # - Ambiguous results → ask for clarification
        # - Max iterations reached without answer → ask for guidance
```

### Step 3: Update Search Agents

Modify `SearchAgent` and `AgenticSearchAgent` to:

1. Accept optional `session_state` parameter to resume
2. Return early with `input_required` when needed
3. Include serializable state in response

```python
# In SearchAgent.search()
async def search(self, query, directory, ..., session_state=None):
    # Restore state if resuming
    if session_state:
        file_index_map = session_state.get("file_index_map", {})
        # ... restore other state

    # Phase 1: SCAN
    scan_result = await self._execute_tool("file_scan", ...)

    if not scan_result:
        return {
            "status": "input_required",
            "input_required": {
                "prompt": f"No files found in '{directory}'. Provide another path?",
                "type": "directory",
                "session_id": session_id,
            },
            "state": self._serialize_state()
        }

    # ... continue phases
```

### Step 4: Add API Endpoints

```python
@router.post("/search")
async def file_search(request: SearchRequest):
    session = SearchSession(app=request.app)
    result = await session.run(
        query=request.query,
        directory=request.directory,
        allowed_directories=request.allowed_directories
    )
    return result

@router.post("/search/resume")
async def resume_search(request: ResumeSearchRequest):
    """Resume a search session with user input."""
    session = SearchSession.from_state(request.session_state)
    result = await session.resume(
        user_input=request.user_input,
        input_type=request.input_type
    )
    return result
```

### Step 5: Input Request Triggers

Define when to request input:

| Condition                       | Input Type  | Prompt                                  |
| ------------------------------- | ----------- | --------------------------------------- |
| No files found in scan          | `directory` | "No files found. Try a different path?" |
| No relevant files after preview | `directory` | "No relevant files. Look elsewhere?"    |
| Max iterations without answer   | `text`      | "Couldn't find answer. More context?"   |
| Ambiguous query                 | `choice`    | "Which topic: A, B, or C?"              |

## Files to Create/Modify

| File                                   | Action                              |
| -------------------------------------- | ----------------------------------- |
| `backends/retrieval/search_session.py` | Create                              |
| `backends/retrieval/search_fs.py`      | Modify (add input_required support) |
| `backends/retrieval/agentic_search.py` | Modify (add input_required support) |
| `backends/inference/classes.py`        | Modify (add response models)        |
| `backends/inference/route.py`          | Modify (add /resume endpoint)       |

## Verification Plan

1. **Test no-files scenario:**

   ```bash
   curl -X POST /v1/search/fs -d '{"query": "...", "directory": "/empty/path"}'
   # Should return status: "input_required"
   ```

2. **Test resume flow:**

   ```bash
   # Get state from input_required response
   curl -X POST /v1/search/fs/resume -d '{
     "session_state": {...},
     "user_input": "/new/path",
     "input_type": "directory"
   }'
   ```

3. **Test workflow integration:**
   - External system receives `input_required` response
   - Prompts user for input
   - Calls `/resume` with user's answer
   - Receives final result

## Usage Example

```python
# Workflow integration pseudocode
async def run_search_workflow(query, directory):
    result = await api.search(query, directory)

    while result.status == "input_required":
        # External system handles the prompt
        user_input = await prompt_user(
            result.input_required.prompt,
            result.input_required.type,
            result.input_required.options
        )

        result = await api.resume_search(
            session_state=result.state,
            user_input=user_input,
            input_type=result.input_required.type
        )

    return result.data.answer
```
