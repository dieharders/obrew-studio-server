# Inference Architecture

How Obrew Studio currently loads and runs [llama.cpp](https://github.com/ggerganov/llama.cpp) for **text inference**, **text embeddings**, and **vision** (multimodal embeddings + multimodal chat).

## TL;DR

Obrew Studio does **not** use `llama-cpp-python` bindings. Instead, the FastAPI backend spawns **native `llama-server` binaries** (shipped in [servers/llama.cpp/](../servers/llama.cpp/)) as child processes and talks to them over **loopback HTTP**. Each purpose — text inference, text embeddings, and vision embeddings — runs as its own `llama-server` subprocess, with dynamically-allocated ports and independent lifecycles managed by thin Python wrapper classes.

This gives us better process isolation, clean GPU resource release on unload, and the ability to swap llama.cpp versions by replacing a binary (no Python rebuild).

## High-level layout

```
FastAPI backend  (Python, asyncio, httpx)
      │
      │   loopback HTTP  (ports 8082–8095, dynamically allocated)
      ▼
┌─────────────────────┬────────────────────────┬────────────────────────┐
│   llama-server      │    llama-server        │     llama-server       │
│   (text inference)  │    (text embedding)    │     (vision embedding) │
│   --jinja           │    --embedding         │     --embedding        │
│   [--mmproj ...]    │    --pooling mean      │     --mmproj ...       │
└─────────────────────┴────────────────────────┴────────────────────────┘
         │                      │                         │
         ▼                      ▼                         ▼
     app.state.llm     app.state.text_embedder   app.state.vision_embedder
```

Only **one** text-inference `llama-server` is alive at a time. The text- and vision-embedding servers are cached on `app.state` and kept alive across requests so you don't pay the model-load cost on every embedding call.

## Port management

All `llama-server` instances acquire a free port from a shared, thread-safe allocator in [backends/core/common.py](../backends/core/common.py) (`allocate_server_port` / `release_server_port`, range **8082–8095**). This prevents collisions between concurrent inference and embedding servers, and guarantees ports are released even when startup raises.

---

## 1. Text inference — `LlamaServer`

### Process spawning and readiness

- Spawned with `asyncio.create_subprocess_exec(...)` (Windows gets `CREATE_NO_WINDOW`).
- `stdout` → `DEVNULL`, `stderr` → `PIPE`. A background task (`_read_logs`) **continuously drains stderr** — without this, Windows' ~4 KB pipe buffer fills and the process blocks on write, stalling HTTP responses.
- The last 20 stderr lines are kept in `_last_stderr_lines` and appended to crash diagnostics.
- Readiness: poll `GET /health` every 1 s until 200, with a **120 s timeout**. If the process dies during startup, the stderr tail is surfaced in the raised error.
- Logs also go to `<app_path>/llama-server.log` for debugging packaged builds that have no console.

### Inference endpoints

Two modes, both streamed from the server as Server-Sent Events and re-emitted through the FastAPI route:

| Mode                               | Obrew `responseMode` | `llama-server` endpoint     | Request shape                                      |
| ---------------------------------- | -------------------- | --------------------------- | -------------------------------------------------- |
| Text completion (prompt-formatted) | `instruct`           | `POST /completion`          | `{ prompt, stream: true, ...gen_params }`          |
| Chat (OpenAI-compatible)           | `chat`               | `POST /v1/chat/completions` | `{ messages: [...], stream: true, ...gen_params }` |

`LlamaServer` always asks the server to stream; if the caller wants a non-streaming response we aggregate tokens server-side in the wrapper and yield a single final payload.

### SSE event protocol

Streaming responses on `/v1/text/inference/generate` are Server-Sent Events. Each event has an `event:` line naming the kind and a `data:` line carrying a JSON payload. Clients should track the **most recent event name** and route the next `data:` payload accordingly.

| Event name             | When emitted                                                                                              | `data` shape                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `FEEDING_PROMPT`       | After the HTTP connection to llama-server is established but before any tokens have been generated.       | _(no data)_                                                        |
| `GENERATING_TOKENS`    | On the first regular content token. Subsequent token frames stream as bare `data:` lines with `{ text }`. | `{ "text": "<token>" }` (per token, when streaming)                |
| `GENERATING_REASONING` | On the first `delta.reasoning_content` token from a reasoning-capable model (requires `--reasoning-format deepseek`, set automatically when the model was loaded with `enable_thinking=True`). Reasoning tokens stream as bare `data:` lines with `{ text }` until regular content begins. | `{ "text": "<reasoning token>" }` (per token, when streaming) |
| `GENERATING_CONTENT`   | Final aggregated payload emitted once at the end of the response.                                         | `{ "text": "<full content>", "reasoning_text"?: "<full reasoning>" }` |

`reasoning_text` is only present on the final `GENERATING_CONTENT` payload when the model actually emitted thinking tokens during streaming.

Non-streaming HTTP requests (`stream: false`) receive the `GENERATING_CONTENT` payload's `data` field as a single JSON body — the response is wrapped in `application/json` with whitespace heartbeats so the connection stays warm during long inference (RFC 8259 permits leading whitespace before a JSON value). Internal callers of `agent.call()` (tool selection, search) bypass the HTTP layer entirely.

### Cancellation

Two-layer abort design:

1. **Per-request `asyncio.Event`** — set from `/v1/text/inference/stop` or when the client disconnects. The generator loop checks it on every SSE line and breaks out.
2. **Server-side stop** — `POST /slots/0?action=erase` tells `llama-server` to drop the in-flight generation immediately instead of buffering the remainder. (Single-slot by default; multi-slot support would need to target the right slot ID.)

The active `httpx.AsyncClient` is force-closed on abort to unblock any hanging stream.

### Unload

`LlamaServer.unload()`:

1. Closes the persistent `httpx.AsyncClient`.
2. Cancels the stderr-draining task.
3. `process.terminate()` → wait 5 s → `process.kill()` if still alive.
4. Releases the port back to the allocator.

On app shutdown, [api_server.py:200-218](../backends/api_server.py#L200-L218) runs `unload()` on `app.state.llm`, `app.state.vision_embedder`, and `app.state.text_embedder` (with a force-kill fallback) so nothing orphans.

---

## 2. Text embeddings — `GGUFEmbedderServer`

### Key differences from `LlamaServer`

- **Synchronous `subprocess.Popen`**, not `asyncio.create_subprocess_exec`. Embedding requests can arrive from background threads (e.g. FastAPI `BackgroundTasks` during memory ingestion) that have no asyncio loop, so the startup path must work without one.
- **Lazy start, eager cache.** The first embedding call starts the server; subsequent calls reuse `app.state.text_embedder`. The embedder module health-checks the cached instance (`process.poll()`) before reuse and rebuilds it if a prior crash left a stale reference.
- 60 s readiness timeout (shorter than inference because the model is smaller).

### Embedding endpoint

```
POST /embeddings
{ "content": "text to embed" }
```

`embed_text()` handles the several response shapes llama.cpp has used over time — `{"embedding": [...]}`, `[{"embedding": [...]}]`, or a bare `[...]` — and returns a single vector. Pooling is configured at startup (`mean` by default), so output is one vector per call.

---

## 3. Vision

Vision shows up in **two** places: embeddings (for image RAG) and chat (multimodal generation). They share the same pattern — load a base model **plus an `mmproj` file** (the multimodal projector that bridges the vision encoder to the LLM) — but live in different wrapper classes.

### 3a. Vision embeddings — `EmbeddingServer` / `ImageEmbedder`

Files:

- [backends/vision/embedding_server.py](../backends/vision/embedding_server.py) — the `llama-server` wrapper.
- [backends/vision/image_embedder.py](../backends/vision/image_embedder.py) — higher-level model discovery, loading, and request normalization.

Default model: **`mradermacher/GME-VARCO-VISION-Embedding-GGUF`**, cached under [vision_embed_models/](../vision_embed_models/). `ImageEmbedder` scans the HF cache layout and pairs the base `.gguf` with a sibling `mmproj*.gguf`.

### Embedding an image

```
POST /embeddings
{
  "content":    "[img-1] <optional prompt>",
  "image_data": [ { "id": 1, "data": "<base64 image>" } ]
}
```

The `[img-N]` placeholder in `content` must match the `id` in `image_data` — this is llama.cpp's multimodal embedding API contract.

Some VLMs (e.g. Qwen2-VL) return **per-token** embeddings instead of a single pooled vector. `_normalize_embedding` ([image_embedder.py:17](../backends/vision/image_embedder.py#L17)) detects that shape and mean-pools into a single vector so downstream vector-DB code sees a consistent format.

Text queries that need to share a vector space with image embeddings use the same vision model via `embed_query_text()` with just `{ "content": "<text>" }`.

### 3b. Vision chat — same `LlamaServer`, loaded with `--mmproj`

Vision chat **reuses the text inference wrapper** — no separate server class. The load route:

1. Unloads any existing `app.state.llm` text model.
2. Instantiates `LlamaServer(..., mmproj_path=...)`, which adds `--mmproj <path>` to the startup command.
3. Stores it back on `app.state.llm`.

Once loaded, image-bearing requests are sent as OpenAI-compatible multimodal messages to `POST /v1/chat/completions`:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } },
        { "type": "text", "text": "What is in this image?" }
      ]
    }
  ]
}
```

Image files are read from disk, base64-encoded, and the MIME type is inferred from the file extension.

---

## Lifecycle summary

| Server                                    | Eager or lazy?              | Cached on                       | Concurrency                         |
| ----------------------------------------- | --------------------------- | ------------------------------- | ----------------------------------- |
| Text inference (`LlamaServer`)            | Eager, on `/v1/text/load`   | `app.state.llm`                 | One at a time (swapped on new load) |
| Vision inference (`LlamaServer` + mmproj) | Eager, on `/v1/vision/load` | `app.state.llm` (replaces text) | One at a time                       |
| Text embeddings (`GGUFEmbedderServer`)    | Lazy, on first call         | `app.state.text_embedder`       | One, reused across all calls        |
| Vision embeddings (`EmbeddingServer`)     | Lazy, on first call         | `app.state.vision_embedder`     | One, reused across all calls        |

Shutdown ([api_server.py](../backends/api_server.py)) tears all three down with `unload()` / `stop()`, falling back to `SIGTERM` on the raw PID if the async path raises, so no `llama-server` process outlives the FastAPI app.

---

[Back to README](../README.md)
