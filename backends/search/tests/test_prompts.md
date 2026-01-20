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

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-vector_request.json"
```

### Web Search (DuckDuckGo)

```bash
curl -X POST "http://localhost:8008/v1/search/web" -H "Content-Type: application/json" -d @"c:\Project Files\brain-dump-ai\obrew-studio-server\backends\search\tests\test_search-web_request.json"
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

### Vector Search - Single Collection

```bash
curl -X POST "http://localhost:8008/v1/search/vector" -H "Content-Type: application/json" -d '{"query": "main concepts", "collection": "my_collection", "allowed_collections": ["my_collection"], "max_extract": 3}'
```
