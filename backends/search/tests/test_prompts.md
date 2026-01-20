Go over your work and verify all implementations have taken place and are feature complete. Fix any bugs or cleanup any unused or unoptimized code. Ensure interfaces are correct and complete everywhere. Then double-check everything again.

## Prerequisites

Load model:

```bash
curl -X POST "http://localhost:8008/v1/text/load" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_load_model.json"
```

## Search Endpoints

### File System Search

```bash
curl -X POST "http://localhost:8008/v1/search/fs/v2" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-fs_request.json"
```

### Vector/Embedding Search

Text request passing collection names:

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-vector_request.json"
```

Text request passing no name:

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-vector_discovery_request.json"
```

Image Request:

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-vector_image_request.json"
```

### Web Search (DuckDuckGo)

```bash
curl -X POST "http://localhost:8008/v1/search/web" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-web_request.json"
```

### Structured Data Search

```bash
curl -X POST "http://localhost:8008/v1/search/structured" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-structured_request.json"
```

## Quick Test Examples

### Web Search - No Domain Restrictions

```bash
curl -X POST "http://localhost:8008/v1/search/web" -H "Content-Type: application/json" -d '{"query": "What is FastAPI?", "max_extract": 2}'
```

### Web Search - Single Website

```bash
curl -X POST "http://localhost:8008/v1/search/web" -H "Content-Type: application/json" -d '{"query": "async await tutorial", "website": "docs.python.org", "max_extract": 2}'
```

### Structured Search - Inline Data

```bash
curl -X POST "http://localhost:8008/v1/search/structured" -H "Content-Type: application/json" -d '{"query": "What is the project status?", "items": [{"content": "Project is on track for Q1 release"}, {"content": "Backend API is 80% complete"}, {"content": "Frontend needs more work on the dashboard"}], "max_extract": 2}'
```

## Stop/Cancel Search

### Stop All Active Searches

```bash
curl -X POST "http://localhost:8008/v1/search/stop"
```

### Stop Specific Search by ID

```bash
curl -X POST "http://localhost:8008/v1/search/stop?search_id=<uuid>"
```

### Test Stop (run search in background, then stop)

Terminal 1 - Start a vector search with collections:

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d '{"query": "What are the key concepts?", "collections": ["my-collection"], "max_extract": 5}'
```

Terminal 2 - Stop it while running:

```bash
curl -X POST "http://localhost:8008/v1/search/stop"
```

Expected response from stop:

```json
{ "success": true, "message": "Stop requested for 1 active searches." }
```

Expected response from cancelled search:

```json
{
  "success": false,
  "message": "Search cancelled.",
  "data": {
    "answer": "Search was cancelled before completion.",
    "sources": [],
    "stats": { "cancelled": true, "cancelled_at_phase": "..." }
  }
}
```

### Test Stop with Discovery Mode (no collections)

Terminal 1 - Start a discovery mode search:

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d '{"query": "Find relevant information", "max_extract": 5}'
```

Terminal 2 - Stop it while running:

```bash
curl -X POST "http://localhost:8008/v1/search/stop"
```
