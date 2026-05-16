"""TTS route handlers for OuteTTS via llama-server."""

import os
import json
import asyncio
import tempfile
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Request, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from starlette.responses import StreamingResponse

from core import classes, common
from core.download_manager import DownloadManager
from inference.classes import (
    CHAT_MODES,
    LoadTextInferenceCall,
    LoadTextInferenceInit,
)
from inference.llama_server import LlamaServer
from inference.route import complete_request

from .classes import (
    LoadTTSRequest,
    LoadTTSResponse,
    LoadedTTSModelResponse,
    DownloadTTSModelRequest,
    DeleteTTSModelRequest,
    TTSGenerateRequest,
    VoiceListResponse,
    DeleteVoiceRequest,
    AddVoiceResponse,
)
from .tts_engine import TTSEngine
from . import helpers


router = APIRouter()

# Curated list of recommended TTS models for the frontend's model browser.
_CURATED_TTS_MODELS = [
    {
        "id": "OuteAI/OuteTTS-1.0-0.6B-GGUF",
        "repoId": "OuteAI/OuteTTS-1.0-0.6B-GGUF",
        "name": "OuteTTS 1.0 (0.6B)",
        "filename": "OuteTTS-1.0-0.6B-Q4_K_M.gguf",
        "description": "Apache 2.0 — recommended default. Voice cloning + multilingual.",
        "license": "Apache-2.0",
        "vocoderRepoId": "ibm-research/DAC.speech.v1.0",
        "vocoderFilename": "weights_24khz_1.5kbps_v1.0.pth",
    },
    {
        "id": "OuteAI/Llama-OuteTTS-1.0-1B-GGUF",
        "repoId": "OuteAI/Llama-OuteTTS-1.0-1B-GGUF",
        "name": "Llama-OuteTTS 1.0 (1B)",
        "filename": "Llama-OuteTTS-1.0-1B-Q4_K_M.gguf",
        "description": "Larger / higher quality. Non-commercial license — read carefully.",
        "license": "CC-BY-NC-SA-4.0",
        "vocoderRepoId": "ibm-research/DAC.speech.v1.0",
        "vocoderFilename": "weights_24khz_1.5kbps_v1.0.pth",
    },
]


def _wrong_model_response(expected: str) -> dict:
    return {
        "success": False,
        "message": (
            f"Wrong model loaded. Call /v1/{expected}/load first (the LLM and TTS "
            "share a single llama-server slot; only one may be active at a time)."
        ),
        "data": None,
    }


def _unload_existing_text(request: Request) -> None:
    """Unload whatever is currently in app.state.llm (text or TTS) before loading TTS."""
    from inference.route import unload_text_inference

    # unload_text_inference is async — schedule via the running loop
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Caller will await this separately; the route handler does that explicitly.
        return
    loop.run_until_complete(unload_text_inference(request))


@router.get("/models")
def get_curated_tts_models():
    """Curated list of recommended TTS models for the frontend browser."""
    return {
        "success": True,
        "message": "Curated TTS models.",
        "data": _CURATED_TTS_MODELS,
    }


@router.get("/installed")
def list_installed_tts_models():
    """List locally-cached TTS models (GGUF + optional vocoder file paths)."""
    try:
        models = common.get_installed_tts_models()
        return {
            "success": True,
            "message": "Installed TTS models.",
            "data": models,
        }
    except Exception as err:
        return {
            "success": False,
            "message": f"Failed to list installed TTS models: {err}",
            "data": [],
        }


@router.post("/download")
def download_tts_model(request: Request, payload: DownloadTTSModelRequest):
    """Start an async download of a TTS model + optional vocoder file.

    Returns task_id(s) immediately. Progress is reported via the shared
    /v1/downloads/progress?task_id=<id> SSE endpoint.
    """
    try:
        app: classes.FastAPIApp = request.app
        download_manager: DownloadManager = app.state.download_manager

        repo_id = payload.repo_id
        filename = payload.filename
        cache_dir = common.app_path(common.TTS_MODELS_CACHE_DIR)
        vocoder_repo_id = payload.vocoder_repo_id
        vocoder_filename = payload.vocoder_filename

        def on_model_complete(task_id: str, file_path: str):
            try:
                [_, repo_revisions] = common.scan_cached_repo(
                    cache_dir=cache_dir, repo_id=repo_id
                )
                actual_path = common.get_cached_blob_path(
                    repo_revisions=repo_revisions, filename=filename
                )
                if not isinstance(actual_path, str):
                    actual_path = file_path
                common.save_tts_model(
                    {
                        "repoId": repo_id,
                        "filename": filename,
                        "savePath": {filename: actual_path},
                    }
                )
                print(
                    f"{common.PRNT_API} TTS model saved: {actual_path}", flush=True
                )
            except Exception as e:
                print(
                    f"{common.PRNT_API} Error saving TTS model metadata: {e}",
                    flush=True,
                )

        def on_model_error(task_id: str, error: Exception):
            print(
                f"{common.PRNT_API} TTS download failed for {repo_id}: {error}",
                flush=True,
            )

        task_id = download_manager.start_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            on_complete=on_model_complete,
            on_error=on_model_error,
        )

        vocoder_task_id: Optional[str] = None
        if vocoder_repo_id and vocoder_filename:
            vocoder_task_id = _start_vocoder_download(
                download_manager=download_manager,
                cache_dir=cache_dir,
                tts_repo_id=repo_id,
                vocoder_repo_id=vocoder_repo_id,
                vocoder_filename=vocoder_filename,
            )
            if vocoder_task_id:
                download_manager.set_secondary_task(task_id, vocoder_task_id)

        data = {"taskId": task_id}
        if vocoder_task_id:
            data["vocoderTaskId"] = vocoder_task_id

        message = f"Download started for {repo_id}/{filename}"
        if vocoder_task_id:
            message += " (vocoder download also started)"

        return {"success": True, "message": message, "data": data}
    except (KeyError, Exception, EnvironmentError, OSError, ValueError) as err:
        print(f"{common.PRNT_API} TTS download error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


def _start_vocoder_download(
    download_manager: DownloadManager,
    cache_dir: str,
    tts_repo_id: str,
    vocoder_repo_id: str,
    vocoder_filename: str,
) -> Optional[str]:
    """Kick off the vocoder weights download and link it to the TTS model entry."""
    try:
        print(
            f"{common.PRNT_API} Starting TTS vocoder download: "
            f"{vocoder_filename} from {vocoder_repo_id}",
            flush=True,
        )

        def on_vocoder_complete(task_id: str, file_path: str):
            try:
                [_, revisions] = common.scan_cached_repo(
                    cache_dir=cache_dir, repo_id=vocoder_repo_id
                )
                actual_path = common.get_cached_blob_path(
                    repo_revisions=revisions, filename=vocoder_filename
                )
                if not isinstance(actual_path, str):
                    actual_path = file_path
                if actual_path:
                    common.save_tts_vocoder_path(tts_repo_id, actual_path)
                    print(
                        f"{common.PRNT_API} Saved TTS vocoder to {actual_path}",
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"{common.PRNT_API} Error saving vocoder path: {e}",
                    flush=True,
                )

        def on_vocoder_error(task_id: str, error: Exception):
            print(
                f"{common.PRNT_API} TTS vocoder download failed: {error}",
                flush=True,
            )

        return download_manager.start_download(
            repo_id=vocoder_repo_id,
            filename=vocoder_filename,
            cache_dir=cache_dir,
            on_complete=on_vocoder_complete,
            on_error=on_vocoder_error,
        )
    except Exception as err:
        print(
            f"{common.PRNT_API} Warning: Could not start TTS vocoder download: {err}",
            flush=True,
        )
        return None


@router.post("/delete")
def delete_tts_model(payload: DeleteTTSModelRequest):
    """Remove a single GGUF file (blob + symlink + metadata) from the local cache."""
    try:
        cache_dir = common.app_path(common.TTS_MODELS_CACHE_DIR)

        try:
            common.check_cached_file_exists(
                cache_dir=cache_dir, repo_id=payload.repoId, filename=payload.filename
            )
            [_, repo_revisions] = common.scan_cached_repo(
                cache_dir=cache_dir, repo_id=payload.repoId
            )
            file_paths = common.get_cached_file_paths(repo_revisions, payload.filename)
            if file_paths:
                blob_path, symlink_path, _ = file_paths
                if os.path.exists(blob_path):
                    os.remove(blob_path)
                if os.path.exists(symlink_path) or os.path.islink(symlink_path):
                    os.remove(symlink_path)
        except Exception as cache_err:
            print(
                f"{common.PRNT_API} TTS cache lookup miss (may be a failed download): {cache_err}",
                flush=True,
            )

        common.delete_tts_model(filename=payload.filename, repo_id=payload.repoId)
        return {
            "success": True,
            "message": f"Deleted TTS model file {payload.filename}.",
            "data": None,
        }
    except (KeyError, Exception) as err:
        print(f"{common.PRNT_API} TTS delete error: {err}", flush=True)
        raise HTTPException(
            status_code=400, detail=f"Something went wrong. Reason: {err}"
        )


@router.post("/load")
async def load_tts(request: Request, payload: LoadTTSRequest) -> dict:
    """Load an OuteTTS model into the shared llama-server slot.

    Unloads any currently loaded model (text or TTS) first.
    """
    app: classes.FastAPIApp = request.app
    try:
        model_id = payload.modelId
        model_path = payload.modelPath or common.get_tts_model_file_path(model_id)
        if not model_path or not os.path.exists(model_path):
            return {
                "success": False,
                "message": (
                    f"TTS model file not found for {model_id}. Download it via "
                    "/v1/tts/download first."
                ),
                "data": None,
            }

        # Unload whatever is currently on app.state.llm (text or TTS)
        from inference.route import unload_text_inference

        if app.state.llm:
            await unload_text_inference(request)
        if app.state.tts_engine:
            try:
                await app.state.tts_engine.unload()
            except Exception:
                pass
            app.state.tts_engine = None

        # Spin up llama-server with the TTS GGUF
        app.state.llm = LlamaServer(
            model_path=model_path,
            model_name="OuteTTS",
            model_id=model_id,
            model_kind="tts",
            response_mode=CHAT_MODES.INSTRUCT.value,
            model_init_kwargs=LoadTextInferenceInit(
                n_ctx=payload.contextSize or 8192,
                n_gpu_layers=999,
                n_threads=None,
                offload_kqv=True,
            ),
            generate_kwargs=LoadTextInferenceCall(),
        )
        started = await app.state.llm.start_server()
        if not started:
            try:
                await app.state.llm.unload()
            except Exception:
                pass
            app.state.llm = None
            return {
                "success": False,
                "message": "Failed to start llama-server for TTS model.",
                "data": None,
            }

        # Resolve vocoder path (downloaded alongside the GGUF if user followed the flow)
        vocoder_path = payload.vocoderPath or common.get_tts_vocoder_path(model_id)

        # Build the TTSEngine and load it
        voices_dir = common.app_path(common.TTS_VOICES_DIR)
        engine = TTSEngine(
            server_host=app.state.llm.base_url,
            voices_dir=voices_dir,
            device=payload.device or "auto",
            vocoder_path=vocoder_path,
        )
        try:
            await engine.load()
        except Exception as e:
            # Roll back the llama-server too
            try:
                await app.state.llm.unload()
            except Exception:
                pass
            app.state.llm = None
            raise

        app.state.tts_engine = engine

        built_in = [v.model_dump() for v in engine.built_in_voices()]
        return {
            "success": True,
            "message": f"TTS model [{model_id}] loaded.",
            "data": {
                "modelId": model_id,
                "device": engine._resolved_device,
                "builtInVoices": built_in,
            },
        }
    except Exception as err:
        print(f"{common.PRNT_API} TTS load error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Failed to load TTS model. Reason: {err}",
            "data": None,
        }


@router.post("/unload")
async def unload_tts(request: Request):
    """Unload the TTS engine and its underlying llama-server."""
    app: classes.FastAPIApp = request.app
    try:
        if app.state.tts_engine:
            try:
                await app.state.tts_engine.unload()
            except Exception as e:
                print(f"{common.PRNT_API} tts_engine.unload error: {e}", flush=True)
            app.state.tts_engine = None
        if app.state.llm and getattr(app.state.llm, "model_kind", "text") == "tts":
            try:
                await app.state.llm.unload()
            except Exception as e:
                print(f"{common.PRNT_API} llm.unload error: {e}", flush=True)
            app.state.llm = None
        return {"success": True, "message": "TTS model ejected.", "data": None}
    except Exception as err:
        print(f"{common.PRNT_API} TTS unload error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Failed to unload TTS model. Reason: {err}",
            "data": None,
        }


@router.get("/model")
def get_loaded_tts_model(request: Request):
    """Return metadata about the currently loaded TTS model."""
    app: classes.FastAPIApp = request.app
    try:
        engine = app.state.tts_engine
        llm = app.state.llm
        if not engine or not llm or getattr(llm, "model_kind", "text") != "tts":
            return {
                "success": False,
                "message": "No TTS model is currently loaded.",
                "data": None,
            }
        built_in = [v.model_dump() for v in engine.built_in_voices()]
        return {
            "success": True,
            "message": f"{llm.model_id} is loaded.",
            "data": {
                "modelId": llm.model_id,
                "device": engine._resolved_device,
                "clonedVoicesCount": len(engine._voices),
                "builtInVoices": built_in,
            },
        }
    except Exception as err:
        return {
            "success": False,
            "message": f"Something went wrong: {err}",
            "data": None,
        }


# ──────────────────────────────────────────────
# Generate endpoints
# ──────────────────────────────────────────────


def _ensure_tts_loaded(app: classes.FastAPIApp):
    llm = app.state.llm
    engine = app.state.tts_engine
    if not engine or not llm or getattr(llm, "model_kind", "text") != "tts":
        raise HTTPException(
            status_code=400,
            detail="No TTS model is loaded. Call /v1/tts/load first.",
        )


async def _queue_and_release(app: classes.FastAPIApp, request: Request):
    """Enqueue the request (mirrors text inference's queueing pattern)."""
    if app.state.request_queue.qsize() > 0:
        raise HTTPException(
            status_code=429,
            detail="Too many requests, please wait.",
        )
    await app.state.request_queue.put(request)


@router.post("/generate")
async def generate(request: Request, payload: TTSGenerateRequest):
    """Synthesize a complete text blob → single WAV response."""
    app: classes.FastAPIApp = request.app
    try:
        _ensure_tts_loaded(app)
        await _queue_and_release(app, request)
        try:
            wav = await app.state.tts_engine.synthesize(
                text=payload.text,
                voice_id=payload.voiceId,
                speaker_id=payload.speakerId,
                speed=payload.speed or 1.0,
                temperature=payload.temperature or 0.4,
            )
        finally:
            await complete_request(app)
        return StreamingResponse(iter([wav]), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as err:
        print(f"{common.PRNT_API} TTS generate error: {err}", flush=True)
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {err}")


@router.post("/generate-stream-out")
async def generate_stream_out(request: Request, payload: TTSGenerateRequest):
    """Text-in (complete) → audio-out (streamed per sentence)."""
    app: classes.FastAPIApp = request.app
    _ensure_tts_loaded(app)
    await _queue_and_release(app, request)

    async def gen():
        try:
            async for chunk in app.state.tts_engine.synthesize_stream(
                text=payload.text,
                voice_id=payload.voiceId,
                speaker_id=payload.speakerId,
                speed=payload.speed or 1.0,
                temperature=payload.temperature or 0.4,
            ):
                if await request.is_disconnected():
                    app.state.tts_engine.request_abort()
                    break
                yield chunk
        finally:
            await complete_request(app)

    return StreamingResponse(gen(), media_type="audio/wav")


@router.post("/generate-stream-in-out")
async def generate_stream_in_out(
    request: Request,
    voiceId: Optional[str] = Query(default=None),
    speakerId: Optional[str] = Query(default="en_female_1"),
    speed: Optional[float] = Query(default=1.0),
    temperature: Optional[float] = Query(default=0.4),
):
    """Text-stream-in (NDJSON) → audio-stream-out.

    Request body is newline-delimited JSON: each line is `{"text": "..."}` (or
    `{"end": true}` to terminate). The server splits accumulated text into
    complete sentences and emits a WAV chunk per sentence.
    """
    app: classes.FastAPIApp = request.app
    _ensure_tts_loaded(app)
    await _queue_and_release(app, request)

    async def ndjson_text_iter() -> AsyncIterator[str]:
        buf = b""
        async for chunk in request.stream():
            if await request.is_disconnected():
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                except Exception:
                    continue
                if obj.get("end"):
                    return
                text = obj.get("text")
                if text:
                    yield text
        # Final flush of any remaining buffered content
        line = buf.strip()
        if line:
            try:
                obj = json.loads(line.decode("utf-8"))
                if not obj.get("end") and obj.get("text"):
                    yield obj["text"]
            except Exception:
                pass

    async def audio_gen():
        try:
            async for chunk in app.state.tts_engine.synthesize_text_stream(
                text_iter=ndjson_text_iter(),
                voice_id=voiceId,
                speaker_id=speakerId,
                speed=speed or 1.0,
                temperature=temperature or 0.4,
            ):
                if await request.is_disconnected():
                    app.state.tts_engine.request_abort()
                    break
                yield chunk
        finally:
            await complete_request(app)

    return StreamingResponse(audio_gen(), media_type="audio/wav")


@router.post("/stop")
async def stop_tts(request: Request):
    """Abort any in-flight streaming synthesis between sentences."""
    app: classes.FastAPIApp = request.app
    if app.state.tts_engine:
        app.state.tts_engine.request_abort()
    return {"success": True, "message": "TTS generation aborted.", "data": None}


# ──────────────────────────────────────────────
# Voice library endpoints
# ──────────────────────────────────────────────


@router.post("/voices/add")
async def add_voice(
    request: Request,
    name: str = Form(...),
    transcript: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
) -> dict:
    """Register a cloned voice from a reference audio upload (10-12 sec wav/mp3)."""
    app: classes.FastAPIApp = request.app
    _ensure_tts_loaded(app)
    try:
        # Save the upload to a temp file so librosa can re-read it.
        suffix = os.path.splitext(file.filename or "ref.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        try:
            info = await asyncio.to_thread(
                app.state.tts_engine.add_voice,
                name,
                tmp_path,
                transcript,
                language,
            )
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        return {
            "success": True,
            "message": f"Voice '{name}' added.",
            "data": info.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as err:
        print(f"{common.PRNT_API} add_voice error: {err}", flush=True)
        return {
            "success": False,
            "message": f"Failed to add voice: {err}",
            "data": None,
        }


@router.get("/voices")
def list_voices(request: Request):
    """List built-in speakers and disk-saved cloned voices."""
    app: classes.FastAPIApp = request.app
    voices = []
    # Built-in voices (engine may or may not be loaded; fall back to static list)
    engine = app.state.tts_engine
    if engine:
        built_in = engine.built_in_voices()
    else:
        # Static fallback so frontends can render before any model is loaded
        built_in = [
            {"voiceId": v, "name": v.replace("_", " ").title(), "language": v.split("_")[0], "gender": v.split("_")[1] if "_" in v else None}
            for v in ("en_male_1", "en_female_1", "en_male_2", "en_female_2")
        ]
        built_in = []  # leave empty; frontend has /v1/tts/model for built-ins
    cloned = []
    if engine:
        cloned = [v.model_dump() for v in engine.list_voices()]
    else:
        # Disk-only enumeration when engine is unloaded
        voices_dir = common.app_path(common.TTS_VOICES_DIR)
        idx = os.path.join(voices_dir, "voices.json")
        try:
            with open(idx, "r") as f:
                disk = json.load(f)
            cloned = [
                {k: vv for k, vv in v.items() if k != "speakerFile"}
                for v in disk.values()
            ]
        except (FileNotFoundError, json.JSONDecodeError):
            cloned = []
    return {
        "success": True,
        "message": f"{len(cloned)} cloned voice(s).",
        "data": cloned,
    }


@router.post("/voices/delete")
def delete_voice(request: Request, payload: DeleteVoiceRequest):
    app: classes.FastAPIApp = request.app
    engine = app.state.tts_engine
    if not engine:
        # Allow deletion-from-disk even when engine is unloaded.
        voices_dir = common.app_path(common.TTS_VOICES_DIR)
        idx = os.path.join(voices_dir, "voices.json")
        try:
            with open(idx, "r") as f:
                disk = json.load(f)
            info = disk.pop(payload.voiceId, None)
            if not info:
                return {
                    "success": False,
                    "message": f"Voice {payload.voiceId} not found.",
                    "data": None,
                }
            for k in ("sourceFile", "speakerFile"):
                p = info.get(k)
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            with open(idx, "w") as f:
                json.dump(disk, f, indent=2)
            return {"success": True, "message": "Voice deleted.", "data": None}
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "success": False,
                "message": "No voices index found.",
                "data": None,
            }

    ok = engine.delete_voice(payload.voiceId)
    return {
        "success": ok,
        "message": "Voice deleted." if ok else f"Voice {payload.voiceId} not found.",
        "data": None,
    }


@router.get("/voices/{voice_id}/sample")
def get_voice_sample(request: Request, voice_id: str):
    """Stream the reference audio clip for a cloned voice."""
    app: classes.FastAPIApp = request.app
    engine = app.state.tts_engine
    path: Optional[str] = None
    if engine:
        path = engine.get_voice_sample_path(voice_id)
    else:
        # Disk-only lookup
        voices_dir = common.app_path(common.TTS_VOICES_DIR)
        idx = os.path.join(voices_dir, "voices.json")
        try:
            with open(idx, "r") as f:
                disk = json.load(f)
            info = disk.get(voice_id)
            if info:
                p = info.get("sourceFile")
                if p and os.path.exists(p):
                    path = p
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    if not path:
        raise HTTPException(status_code=404, detail="Voice sample not found.")
    return FileResponse(path, media_type="audio/wav")
