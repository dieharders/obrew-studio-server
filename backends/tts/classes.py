from typing import List, Literal, Optional
from pydantic import BaseModel


class LoadTTSRequest(BaseModel):
    modelId: str
    modelPath: Optional[str] = None  # resolved from modelId if absent
    vocoderPath: Optional[str] = None  # resolved from modelId if absent
    device: Optional[Literal["auto", "cuda", "mps", "cpu"]] = "auto"
    contextSize: Optional[int] = 8192


class LoadTTSResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None  # {modelId, device, builtInVoices}


class DownloadTTSModelRequest(BaseModel):
    repo_id: str
    filename: str
    vocoder_repo_id: Optional[str] = None
    vocoder_filename: Optional[str] = None


class DeleteTTSModelRequest(BaseModel):
    repoId: str
    filename: str


class TTSModelInstall(BaseModel):
    repoId: str
    filename: Optional[str] = None
    savePath: dict
    vocoderPath: Optional[str] = None


class TTSGenerateRequest(BaseModel):
    text: str
    voiceId: Optional[str] = None  # None = use default built-in speaker
    speakerId: Optional[str] = "en_female_1"
    speed: Optional[float] = 1.0
    temperature: Optional[float] = 0.4


class VoiceInfo(BaseModel):
    voiceId: str
    name: str
    createdAt: str
    sampleRate: int
    sourceFile: str
    language: Optional[str] = None


class VoiceListResponse(BaseModel):
    success: bool
    message: str
    data: List[VoiceInfo]


class DeleteVoiceRequest(BaseModel):
    voiceId: str


class BuiltInVoice(BaseModel):
    voiceId: str  # e.g. "en_female_1"
    name: str
    language: Optional[str] = None  # ISO code, e.g. "en", "zh", "es"
    gender: Optional[str] = None  # "male" / "female" / None


class LoadedTTSModelResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None  # {modelId, device, clonedVoicesCount, builtInVoices}


class AddVoiceResponse(BaseModel):
    success: bool
    message: str
    data: Optional[VoiceInfo] = None
