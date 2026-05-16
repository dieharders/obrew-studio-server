"""TTSEngine: pure-Python orchestrator over an OuteTTS llama-server + DAC vocoder.

The llama-server subprocess is owned by the shared `app.state.llm` slot and is
spawned/torn down by the /v1/tts/load and /v1/tts/unload route handlers. This
engine just wraps the `outetts.Interface` HTTP client and the in-process DAC
vocoder, plus the on-disk cloned-voice library.
"""

from __future__ import annotations

import os
import json
import asyncio
import time
import shutil
import gc
from typing import AsyncIterator, Iterator, List, Optional

from core import common
from .classes import BuiltInVoice, VoiceInfo
from . import helpers


LOG_PREFIX = f"{common.bcolors.OKCYAN}[TTS-ENGINE]{common.bcolors.ENDC}"

# Voice cloning reference settings (per OuteTTS 1.0 model card)
REFERENCE_TARGET_SR = 24000
REFERENCE_MAX_SECONDS = 12.0
DEFAULT_LANGUAGE = "en"


class TTSEngine:
    """Wraps outetts.Interface + DAC vocoder + cloned-voice library.

    Does NOT own the llama-server subprocess; the route handler manages that
    via the shared LlamaServer at app.state.llm.
    """

    def __init__(
        self,
        server_host: str,
        voices_dir: str,
        device: str = "auto",
        vocoder_path: Optional[str] = None,
    ):
        self.server_host = server_host.rstrip("/")
        self.voices_dir = voices_dir
        self.device = device
        self.vocoder_path = vocoder_path

        os.makedirs(self.voices_dir, exist_ok=True)
        self._voices_index_path = os.path.join(self.voices_dir, "voices.json")
        self._voices: dict = self._load_voices_index()

        # Resolved-at-load values
        self._resolved_device: Optional[str] = None
        self._interface = None  # outetts.Interface
        self._abort_event = asyncio.Event()
        self._is_loaded = False

    # ──────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────

    async def load(self):
        """Build the outetts.Interface and load the DAC vocoder."""
        if self._is_loaded:
            return

        # Resolve device
        self._resolved_device = self._resolve_device(self.device)
        print(
            f"{LOG_PREFIX} Loading TTS engine on device={self._resolved_device}, server={self.server_host}",
            flush=True,
        )

        try:
            import outetts  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "outetts package not installed. Install with `pip install outetts`."
            ) from e

        # Build ModelConfig directly — auto_config raises NotImplementedError for LLAMACPP_SERVER.
        # Use Interface V3 for OuteTTS 1.0.
        try:
            interface_version = outetts.InterfaceVersion.V3
        except AttributeError:
            interface_version = getattr(outetts, "InterfaceVersion", None)

        config_kwargs = dict(
            backend=outetts.Backend.LLAMACPP_SERVER,
            server_host=self.server_host,
        )
        if interface_version is not None and not isinstance(interface_version, type(None)):
            config_kwargs["interface_version"] = (
                interface_version.V3
                if hasattr(interface_version, "V3")
                else interface_version
            )

        model_config = outetts.ModelConfig(**config_kwargs)

        # Building the Interface triggers the DAC vocoder download / load.
        # We run it off the event loop so the FastAPI server stays responsive.
        self._interface = await asyncio.to_thread(outetts.Interface, model_config)
        self._is_loaded = True
        print(f"{LOG_PREFIX} TTS engine ready", flush=True)

    async def unload(self):
        """Drop the outetts.Interface and free GPU memory."""
        try:
            self._interface = None
            self._is_loaded = False
            # Best-effort GPU memory cleanup
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, "mps") and torch.backends.mps.is_available():
                    try:
                        torch.mps.empty_cache()
                    except Exception:
                        pass
            except Exception:
                pass
            gc.collect()
        except Exception as e:
            print(f"{LOG_PREFIX} Error during unload: {e}", flush=True)

    def is_loaded(self) -> bool:
        return self._is_loaded

    # ──────────────────────────────────────────────
    # Synthesis
    # ──────────────────────────────────────────────

    def _resolve_speaker(self, voice_id: Optional[str], speaker_id: Optional[str]):
        """Pick a Speaker object for outetts.

        Priority:
            1) `voice_id` is one of our saved cloned voices  →  load saved Speaker
            2) `speaker_id` is one of outetts built-in default speakers  →  load_default_speaker
            3) Fallback: outetts' default speaker for the language
        """
        if self._interface is None:
            raise RuntimeError("TTS engine is not loaded")

        # Saved cloned voice
        if voice_id and voice_id in self._voices:
            speaker_path = self._voices[voice_id].get("speakerFile")
            if speaker_path and os.path.exists(speaker_path):
                try:
                    return self._interface.load_speaker(path=speaker_path)
                except Exception as e:
                    print(
                        f"{LOG_PREFIX} Failed to load cloned voice {voice_id}: {e}",
                        flush=True,
                    )

        # Built-in speaker
        if speaker_id:
            try:
                return self._interface.load_default_speaker(speaker_id)
            except Exception as e:
                print(
                    f"{LOG_PREFIX} Built-in speaker '{speaker_id}' unavailable: {e}",
                    flush=True,
                )

        # Final fallback — let outetts pick a default for English
        try:
            return self._interface.load_default_speaker("en_female_1")
        except Exception:
            return None

    def _synthesize_sync(
        self,
        text: str,
        voice_id: Optional[str],
        speaker_id: Optional[str],
        temperature: float,
    ) -> bytes:
        """Run a single synthesis call and return WAV bytes.

        Runs on a worker thread (called via asyncio.to_thread).
        """
        if self._interface is None:
            raise RuntimeError("TTS engine is not loaded")

        import outetts  # type: ignore
        import tempfile
        import os as _os

        speaker = self._resolve_speaker(voice_id, speaker_id)

        # Per-request gen config. repetition_range=64 is REQUIRED per model card —
        # full-context repetition penalty range breaks audio.
        gen_config_kwargs = dict(
            text=text,
            generation_type=getattr(outetts.GenerationType, "CHUNKED", None)
            or getattr(outetts.GenerationType, "REGULAR", None),
            speaker=speaker,
            sampler_config=outetts.SamplerConfig(
                temperature=temperature,
                repetition_penalty=1.1,
                repetition_range=64,
                top_k=40,
                top_p=0.95,
                min_p=0.05,
            ),
        )
        # Some outetts versions wrap config in GenerationConfig
        try:
            gen_config = outetts.GenerationConfig(**gen_config_kwargs)
            output = self._interface.generate(config=gen_config)
        except (TypeError, AttributeError):
            output = self._interface.generate(**gen_config_kwargs)

        # Save WAV via outetts' built-in save_audio; fall back to numpy conversion
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = _os.path.join(tmp, "out.wav")
            try:
                if hasattr(output, "save"):
                    output.save(wav_path)
                else:
                    # Best-effort: assume `output` exposes .audio (numpy) and .sr (int)
                    audio = getattr(output, "audio", None)
                    sr = getattr(output, "sr", helpers.DEFAULT_SAMPLE_RATE) or helpers.DEFAULT_SAMPLE_RATE
                    if audio is None:
                        raise RuntimeError("outetts output has no audio data")
                    wav_bytes = helpers.numpy_to_wav_bytes(audio, sample_rate=sr)
                    return wav_bytes
            except Exception as e:
                # If we can't save through outetts, try the numpy path as a last resort
                audio = getattr(output, "audio", None)
                if audio is not None:
                    sr = getattr(output, "sr", helpers.DEFAULT_SAMPLE_RATE) or helpers.DEFAULT_SAMPLE_RATE
                    return helpers.numpy_to_wav_bytes(audio, sample_rate=sr)
                raise RuntimeError(f"Failed to extract audio from outetts output: {e}")

            with open(wav_path, "rb") as f:
                return f.read()

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speaker_id: Optional[str] = "en_female_1",
        speed: float = 1.0,
        temperature: float = 0.4,
    ) -> bytes:
        """Synthesize a complete text blob → WAV bytes (one-shot)."""
        if not self._is_loaded:
            raise RuntimeError("TTS engine is not loaded")
        wav = await asyncio.to_thread(
            self._synthesize_sync, text, voice_id, speaker_id, temperature
        )
        return wav

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        speaker_id: Optional[str] = "en_female_1",
        speed: float = 1.0,
        temperature: float = 0.4,
        language: str = DEFAULT_LANGUAGE,
    ) -> AsyncIterator[bytes]:
        """Split `text` into sentences, synthesize each in turn, yield PCM frames.

        First chunk is a streaming WAV header (data_size=0xFFFFFFFF) so clients
        can begin playback before all sentences have been generated.
        """
        if not self._is_loaded:
            raise RuntimeError("TTS engine is not loaded")

        self._abort_event.clear()
        sentences = helpers.segment_sentences(text, language=language)
        if not sentences:
            sentences = [text]

        yield helpers.make_wav_header()

        for sentence in sentences:
            if self._abort_event.is_set():
                break
            try:
                wav_bytes = await asyncio.to_thread(
                    self._synthesize_sync,
                    sentence,
                    voice_id,
                    speaker_id,
                    temperature,
                )
                pcm = helpers.strip_wav_header(wav_bytes)
                if pcm:
                    yield pcm
            except Exception as e:
                print(
                    f"{LOG_PREFIX} synthesize_stream sentence failed: {e}",
                    flush=True,
                )
                # Continue with the next sentence rather than aborting the whole stream.

    async def synthesize_text_stream(
        self,
        text_iter: AsyncIterator[str],
        voice_id: Optional[str] = None,
        speaker_id: Optional[str] = "en_female_1",
        speed: float = 1.0,
        temperature: float = 0.4,
        language: str = DEFAULT_LANGUAGE,
    ) -> AsyncIterator[bytes]:
        """Accept a stream of text fragments and emit audio per completed sentence.

        Yields:
            streaming WAV header (once), then raw int16 PCM frames per sentence.
        """
        if not self._is_loaded:
            raise RuntimeError("TTS engine is not loaded")

        self._abort_event.clear()
        yield helpers.make_wav_header()

        buffer = ""
        async for fragment in text_iter:
            if self._abort_event.is_set():
                return
            if not fragment:
                continue
            buffer += fragment
            sentences, remainder = helpers.pysbd_split_complete(buffer, language=language)
            buffer = remainder
            for sentence in sentences:
                if self._abort_event.is_set():
                    return
                try:
                    wav_bytes = await asyncio.to_thread(
                        self._synthesize_sync,
                        sentence,
                        voice_id,
                        speaker_id,
                        temperature,
                    )
                    pcm = helpers.strip_wav_header(wav_bytes)
                    if pcm:
                        yield pcm
                except Exception as e:
                    print(
                        f"{LOG_PREFIX} synthesize_text_stream sentence failed: {e}",
                        flush=True,
                    )

        # Flush trailing remainder when the input stream ends.
        if buffer.strip() and not self._abort_event.is_set():
            try:
                wav_bytes = await asyncio.to_thread(
                    self._synthesize_sync,
                    buffer.strip(),
                    voice_id,
                    speaker_id,
                    temperature,
                )
                pcm = helpers.strip_wav_header(wav_bytes)
                if pcm:
                    yield pcm
            except Exception as e:
                print(f"{LOG_PREFIX} flush remainder failed: {e}", flush=True)

    def request_abort(self):
        """Signal in-flight streaming generators to stop between sentences."""
        self._abort_event.set()

    # ──────────────────────────────────────────────
    # Voice library
    # ──────────────────────────────────────────────

    def _load_voices_index(self) -> dict:
        try:
            with open(self._voices_index_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_voices_index(self):
        with open(self._voices_index_path, "w") as f:
            json.dump(self._voices, f, indent=2)

    def add_voice(
        self,
        name: str,
        reference_audio_path: str,
        transcript: Optional[str] = None,
        language: Optional[str] = None,
    ) -> VoiceInfo:
        """Register a cloned voice from a reference audio clip.

        The reference is resampled to 24 kHz mono and trimmed to ~12 s. The
        resulting Speaker object is serialized to disk so it survives restarts.
        """
        if self._interface is None:
            raise RuntimeError("TTS engine is not loaded; load a TTS model first")

        from nanoid import generate as nanoid_generate  # type: ignore

        voice_id = f"v_{nanoid_generate(size=10)}"

        # Resample reference audio to 24 kHz mono and trim length.
        try:
            import librosa  # type: ignore
            import soundfile as sf  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "librosa + soundfile are required for voice cloning. "
                "Install with `pip install librosa soundfile`."
            ) from e

        audio, _ = librosa.load(
            reference_audio_path, sr=REFERENCE_TARGET_SR, mono=True
        )
        max_samples = int(REFERENCE_MAX_SECONDS * REFERENCE_TARGET_SR)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]

        ref_wav_path = os.path.join(self.voices_dir, f"{voice_id}_ref.wav")
        sf.write(ref_wav_path, audio, REFERENCE_TARGET_SR)

        # Build the speaker via outetts; save_speaker writes a JSON encoding of
        # the embedding/codes which load_speaker can read back later.
        speaker_path = os.path.join(self.voices_dir, f"{voice_id}.speaker.json")
        try:
            speaker = self._interface.create_speaker(
                audio_path=ref_wav_path,
                transcript=transcript,
            )
            self._interface.save_speaker(speaker=speaker, path=speaker_path)
        except TypeError:
            # Some outetts versions: create_speaker takes positional arg
            try:
                speaker = self._interface.create_speaker(ref_wav_path)
                self._interface.save_speaker(speaker, speaker_path)
            except Exception as e:
                # Cleanup on failure
                for p in (ref_wav_path, speaker_path):
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                raise RuntimeError(f"Failed to create speaker: {e}") from e

        info = {
            "voiceId": voice_id,
            "name": name,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sampleRate": REFERENCE_TARGET_SR,
            "sourceFile": ref_wav_path,
            "speakerFile": speaker_path,
            "language": language,
        }
        self._voices[voice_id] = info
        self._save_voices_index()
        return VoiceInfo(**{k: v for k, v in info.items() if k != "speakerFile"})

    def list_voices(self) -> List[VoiceInfo]:
        result: List[VoiceInfo] = []
        for v in self._voices.values():
            try:
                result.append(VoiceInfo(**{k: vv for k, vv in v.items() if k != "speakerFile"}))
            except Exception:
                continue
        return result

    def delete_voice(self, voice_id: str) -> bool:
        info = self._voices.pop(voice_id, None)
        if not info:
            return False
        for path_key in ("sourceFile", "speakerFile"):
            p = info.get(path_key)
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception as e:
                    print(f"{LOG_PREFIX} Could not remove {p}: {e}", flush=True)
        self._save_voices_index()
        return True

    def get_voice_sample_path(self, voice_id: str) -> Optional[str]:
        info = self._voices.get(voice_id)
        if not info:
            return None
        p = info.get("sourceFile")
        return p if p and os.path.exists(p) else None

    def built_in_voices(self) -> List[BuiltInVoice]:
        """Enumerate outetts default speakers as structured BuiltInVoice entries."""
        if self._interface is None:
            # Surface a small static fallback so the frontend can render *something*
            # before the engine is loaded.
            return [
                BuiltInVoice(**helpers.parse_builtin_voice_id(v))
                for v in (
                    "en_male_1",
                    "en_female_1",
                    "en_male_2",
                    "en_female_2",
                )
            ]
        ids: List[str] = []
        for attr in ("default_speakers", "list_default_speakers"):
            fn = getattr(self._interface, attr, None)
            try:
                if callable(fn):
                    out = fn()
                    if isinstance(out, dict):
                        ids = list(out.keys())
                    elif isinstance(out, (list, tuple)):
                        ids = list(out)
                    break
                elif fn is not None:
                    ids = list(fn)
                    break
            except Exception:
                continue
        if not ids:
            ids = ["en_male_1", "en_female_1", "en_male_2", "en_female_2"]
        return [BuiltInVoice(**helpers.parse_builtin_voice_id(v)) for v in ids]

    # ──────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────

    def _resolve_device(self, device: str) -> str:
        """Map 'auto' to the best available torch device."""
        if device != "auto":
            return device
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
