"""TTS helpers: WAV header construction, sentence segmentation, audio I/O utilities."""

import io
import wave
import struct
import base64
from typing import List, Tuple

# Default OuteTTS output format
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_CHANNELS = 1
DEFAULT_BITS = 16

# "Unknown length" sentinel used in streaming WAV headers. Both major browsers
# and ffmpeg accept this so progressive playback works for chunked responses.
_UNKNOWN_LENGTH = 0xFFFFFFFF


def make_wav_header(
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    bits: int = DEFAULT_BITS,
    data_size: int = _UNKNOWN_LENGTH,
) -> bytes:
    """Build a 44-byte RIFF WAV header.

    For streaming responses pass data_size=0xFFFFFFFF (the default) so the
    consumer doesn't require Content-Length and can play progressively.
    """
    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    # Total RIFF chunk size = data_size + 36 (header bytes after the RIFF size field)
    riff_size = (
        _UNKNOWN_LENGTH if data_size == _UNKNOWN_LENGTH else data_size + 36
    )
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)  # PCM fmt chunk size
        + struct.pack("<H", 1)  # PCM format
        + struct.pack("<H", channels)
        + struct.pack("<I", sample_rate)
        + struct.pack("<I", byte_rate)
        + struct.pack("<H", block_align)
        + struct.pack("<H", bits)
        + b"data"
        + struct.pack("<I", data_size)
    )


def strip_wav_header(wav_bytes: bytes) -> bytes:
    """Strip the RIFF/fmt/data header(s) from a WAV byte string, returning raw PCM.

    Handles both the canonical 44-byte header and longer headers that include
    extra chunks (LIST/INFO etc) by walking chunks until 'data' is found.
    """
    if len(wav_bytes) < 12 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        return wav_bytes  # Not a WAV; pass through

    offset = 12
    while offset + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[offset : offset + 4]
        chunk_size = struct.unpack("<I", wav_bytes[offset + 4 : offset + 8])[0]
        if chunk_id == b"data":
            return wav_bytes[offset + 8 : offset + 8 + chunk_size]
        offset += 8 + chunk_size
        # Chunks are word-aligned
        if chunk_size % 2 == 1:
            offset += 1
    return b""


def pcm_to_wav_bytes(
    pcm: bytes,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    channels: int = DEFAULT_CHANNELS,
    bits: int = DEFAULT_BITS,
) -> bytes:
    """Wrap raw int16 LE PCM bytes in a fully-sized WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(bits // 8)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def numpy_to_wav_bytes(audio, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """Convert a numpy float waveform to 16-bit PCM WAV bytes."""
    import numpy as np  # local import: numpy is only needed at runtime

    audio = np.asarray(audio).squeeze()
    # Clip and convert to int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16).tobytes()
    return pcm_to_wav_bytes(pcm, sample_rate=sample_rate)


def numpy_to_pcm_bytes(audio) -> bytes:
    """Convert a numpy float waveform to raw 16-bit PCM bytes (no header)."""
    import numpy as np

    audio = np.asarray(audio).squeeze()
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


# Sentence terminators we treat as "complete" when streaming text in.
_TERMINATORS = {".", "!", "?", "\n", "。", "！", "？"}


def pysbd_split_complete(buffer: str, language: str = "en") -> Tuple[List[str], str]:
    """Split a text buffer into (complete_sentences, trailing_remainder).

    The trailing remainder is whatever follows the last sentence-ending punctuation,
    so the caller can feed more text into it before re-splitting. Used by the
    text-stream-in mode so partial LLM outputs don't get cut mid-sentence.
    """
    if not buffer:
        return [], ""

    # Find the index of the last terminator character.
    last_term = -1
    for i in range(len(buffer) - 1, -1, -1):
        if buffer[i] in _TERMINATORS:
            last_term = i
            break

    if last_term == -1:
        return [], buffer

    # Split the buffered portion up to and including the last terminator.
    head = buffer[: last_term + 1]
    remainder = buffer[last_term + 1 :]

    # Use pysbd for the actual sentence segmentation inside `head`.
    try:
        import pysbd  # local import

        seg = pysbd.Segmenter(language=language, clean=False)
        sentences = [s.strip() for s in seg.segment(head) if s.strip()]
    except Exception:
        # Fallback: split on terminator boundaries.
        sentences = [s.strip() for s in head.replace("\n", ". ").split(". ") if s.strip()]

    return sentences, remainder


def segment_sentences(text: str, language: str = "en") -> List[str]:
    """Split a complete text blob into sentences via pysbd."""
    try:
        import pysbd

        seg = pysbd.Segmenter(language=language, clean=False)
        return [s.strip() for s in seg.segment(text) if s.strip()]
    except Exception:
        return [s.strip() for s in text.split(". ") if s.strip()]


def decode_base64_audio(b64: str, out_path: str) -> str:
    """Decode a base64-encoded audio blob and write it to disk; returns the path."""
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path


def parse_builtin_voice_id(voice_id: str) -> dict:
    """Best-effort decode an OuteTTS speaker id like 'en_female_1' into structured metadata."""
    parts = voice_id.split("_")
    language = parts[0] if parts else None
    gender = None
    if len(parts) >= 2 and parts[1] in ("male", "female"):
        gender = parts[1]
    name = voice_id.replace("_", " ").title()
    return {
        "voiceId": voice_id,
        "name": name,
        "language": language,
        "gender": gender,
    }
