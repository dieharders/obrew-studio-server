# Search API

The Search API provides agentic, multi-phase search capabilities across different data sources. All search endpoints use a unified architecture that combines LLM-guided selection with progressive content extraction.

## Architecture Overview

The search system follows a **7-phase agentic search loop**:

```
1. DISCOVER    - Find items in scope (directory, collection, search query)
2. LLM SELECT  - Use LLM to select relevant items for preview
3. PREVIEW     - Get preview content for selected items
4. LLM SELECT  - Use LLM to select items for full extraction
5. EXTRACT     - Get full content for selected items
6. EXPAND      - Search additional scopes if needed (optional)
7. SYNTHESIZE  - Generate final answer from context
```

This architecture is implemented in the `AgenticSearch` orchestrator which works with pluggable `SearchProvider` implementations for different data sources.

## Prerequisites

All search endpoints require:
- A loaded LLM model (the model synthesizes answers from retrieved content)

## Endpoints

All search endpoints are mounted under `/v1/search/`.

---

### POST `/v1/search/vector`

Perform agentic search over ChromaDB vector collections.

**Use cases:**
- Semantic search over embedded documents
- Knowledge base queries
- RAG (Retrieval Augmented Generation)

**Request Body:**

```json
{
  "query": "string",           // Required: The search query
  "collections": ["string"],   // Optional: Collection names to search (null = discover all)
  "top_k": 50,                 // Optional: Max chunks to retrieve per collection
  "max_preview": 10,           // Optional: Max items to preview
  "max_extract": 3,            // Optional: Max items to extract full content from
  "auto_expand": true          // Optional: Search additional collections if needed
}
```

**Discovery Mode:** If `collections` is `null` or empty, the endpoint operates in discovery mode - listing all available collections and letting the LLM select which to search.

**Example:**

```json
{
  "query": "What are the key findings from the research?",
  "collections": ["research_papers", "technical_docs"],
  "top_k": 50,
  "max_preview": 10,
  "max_extract": 3
}
```

---

### POST `/v1/search/web`

Perform agentic web search using DuckDuckGo.

**Use cases:**
- Real-time web information retrieval
- Research queries
- Documentation lookups

**Request Body:**

```json
{
  "query": "string",           // Required: The search query
  "website": ["string"],       // Optional: Domain filter (see below)
  "max_pages": 10,             // Optional: Max pages to fetch content from
  "max_preview": 10,           // Optional: Max URLs to preview
  "max_extract": 3             // Optional: Max pages to extract full content from
}
```

**Domain Filtering (`website` parameter):**
- `null` or `[]` - Search all domains
- `["example.com"]` - Single site search (uses `site:` operator)
- `["docs.python.org", "stackoverflow.com"]` - Whitelist multiple domains

**Example:**

```json
{
  "query": "Python asyncio best practices",
  "website": ["docs.python.org", "stackoverflow.com"],
  "max_pages": 10,
  "max_preview": 10,
  "max_extract": 3
}
```

**Rate Limiting:** The web provider includes built-in rate limiting (0.5s between requests) and concurrency limits (max 3 concurrent page fetches) to avoid overwhelming target servers.

---

### POST `/v1/search/fs`

Perform agentic file system search.

**Use cases:**
- Local document search
- Finding files by content
- Searching project files

**Request Body:**

```json
{
  "query": "string",                    // Required: The search query
  "directory": "string",                // Required: Directory to search in
  "allowed_directories": ["string"],    // Required: Whitelist of accessible directories
  "file_patterns": [".pdf", ".docx"],   // Optional: File extension filter
  "max_files_preview": 10,              // Optional: Max files to preview
  "max_files_parse": 3,                 // Optional: Max files to fully parse
  "max_iterations": 3,                  // Optional: Max directories for expansion
  "auto_expand": true                   // Optional: Search additional directories
}
```

**Security:** The `allowed_directories` whitelist ensures the agent can only access permitted paths. The search directory must be in or under an allowed directory.

**Example:**

```json
{
  "query": "Find all documents about quarterly sales reports",
  "directory": "/documents/reports",
  "allowed_directories": ["/documents/reports", "/documents/archives"],
  "file_patterns": [".pdf", ".docx"],
  "max_files_preview": 10,
  "max_files_parse": 3
}
```

---

### POST `/v1/search/structured`

Perform agentic search over client-provided structured data.

**Use cases:**
- Searching conversation history
- Finding relevant project metadata
- Searching workflow data
- Any ephemeral data the frontend has that the server cannot access

**Request Body:**

```json
{
  "query": "string",           // Required: The search query
  "items": [                   // Required: Array of items to search
    {
      "id": "string",          // Optional: Auto-generated if not provided
      "name": "string",        // Optional: Defaults to "Item {index}"
      "content": "any",        // Required: String, object, array, or primitive
      "metadata": {}           // Optional: Additional metadata
    }
  ],
  "max_preview": 10,           // Optional: Max items to preview
  "max_extract": 3,            // Optional: Max items to extract from
  "group_by": "string",        // Optional: Metadata field to group items by
  "auto_expand": false         // Optional: Search additional groups
}
```

**Limits:**
- Maximum 1000 items per request
- Maximum 50MB total payload size
- Maximum content nesting depth of 10

**Example:**

```json
{
  "query": "What did we decide about authentication?",
  "items": [
    {
      "id": "msg-001",
      "name": "Alice",
      "content": "I think we should use JWT tokens for authentication.",
      "metadata": {"channel": "engineering"}
    },
    {
      "id": "msg-002",
      "name": "Bob",
      "content": {"text": "Agreed. We can use jose library.", "attachments": []},
      "metadata": {"channel": "engineering"}
    }
  ],
  "max_preview": 10,
  "max_extract": 3,
  "group_by": "channel",
  "auto_expand": true
}
```

---

### POST `/v1/search/stop`

Stop an active search operation.

**Request Body:**

```json
{
  "search_id": "string"   // Optional: Specific search ID to stop (null = stop all)
}
```

**Response:**

```json
{
  "success": true,
  "message": "Stop requested for search {id}."
}
```

---

## Response Format

All search endpoints return a unified `SearchResult` structure:

```json
{
  "success": true,
  "message": "Search completed successfully.",
  "data": {
    "answer": "string",           // LLM-generated answer based on retrieved content
    "sources": [                  // List of sources used to generate the answer
      {
        "id": "string",
        "type": "string",         // "filesystem" | "vector" | "web" | "structured"
        "name": "string",
        "snippet": "string"       // Preview of source content
      }
    ],
    "query": "string",            // Original query
    "search_type": "string",      // Type of search performed
    "stats": {                    // Search statistics
      "scopes_searched": [],
      "items_discovered": 0,
      "items_previewed": 0,
      "items_extracted": 0
    },
    "tool_logs": [                // Phase-by-phase execution logs
      {
        "phase": "discover",
        "scope": "...",
        "items_found": 10
      }
    ]
  }
}
```

**Error Response:**

```json
{
  "success": false,
  "message": "Error description",
  "data": null
}
```

**Cancelled Response:**

```json
{
  "success": false,
  "message": "Search cancelled.",
  "data": {
    "answer": "Search was cancelled before completion.",
    "sources": [],
    "query": "...",
    "search_type": "...",
    "stats": {
      "cancelled": true,
      "cancelled_at_phase": "preview"
    },
    "tool_logs": [...]
  }
}
```

---

## Client Disconnect Handling

All search endpoints support automatic cancellation when the client disconnects. The search loop checks for disconnection after each phase and returns a cancelled result if the client is no longer connected.

---

## Adding Custom Search Providers

The search architecture is extensible. To add a new search provider:

1. Create a class that implements the `SearchProvider` protocol in `backends/search/providers/`:

```python
from backends.search.harness import SearchProvider, SearchItem

class MyProvider(SearchProvider):
    async def discover(self, scope: str, **kwargs) -> List[SearchItem]:
        """Find items in the given scope."""
        pass

    async def preview(self, items: List[SearchItem]) -> List[SearchItem]:
        """Get preview content for items."""
        pass

    async def extract(self, items: List[SearchItem]) -> List[Dict[str, str]]:
        """Extract full content from items."""
        pass

    def get_expandable_scopes(self, current_scope: str) -> List[str]:
        """Return additional scopes to search."""
        pass
```

2. Create a route in `backends/search/route.py` that uses `AgenticSearch` with your provider.

3. Add request/response models in `backends/search/classes.py`.

[Back to README](../README.md)
