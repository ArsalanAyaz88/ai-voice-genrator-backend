"""FastAPI entry point for Kokoro TTS Local."""

from __future__ import annotations

import io
from typing import List

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from models import (
    LANGUAGE_CODES,
    build_model,
    generate_speech,
    get_language_code_from_voice,
    list_available_voices,
)

APP_TITLE = "Kokoro TTS Local API"
APP_DESCRIPTION = "REST API for generating speech with Kokoro voices."
SAMPLE_RATE = 24000
MIN_SPEED = 0.1
MAX_SPEED = 3.0
MAX_TEXT_LENGTH = 10_000

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    voice: str = Field(
        "af_bella",
        description="Voice name (see /voices). Either 'af_bella' or 'af_bella.pt'",
    )
    speed: float = Field(
        1.0,
        ge=MIN_SPEED,
        le=MAX_SPEED,
        description="Playback speed multiplier",
    )
    device: str = Field(
        "cpu",
        pattern="^(cpu|cuda)$",
        description="Device used for inference",
    )
    return_phonemes: bool = Field(
        False,
        description="Include phoneme sequence in the X-Phonemes response header",
    )


def _standardize_voice_name(voice: str) -> str:
    voice = voice.strip()
    return voice[:-3] if voice.endswith(".pt") else voice


def _get_pipeline(voice_name: str, device: str) -> torch.nn.Module:
    lang_code = get_language_code_from_voice(voice_name)
    return build_model(None, device, lang_code=lang_code)


@app.get("/health")
def health_check() -> dict:
    """Simple health endpoint for uptime monitoring."""
    voices = list_available_voices()
    return {
        "status": "ok",
        "voices_cached": len(voices),
        "sample_rate": SAMPLE_RATE,
        "min_speed": MIN_SPEED,
        "max_speed": MAX_SPEED,
    }


@app.get("/voices")
def list_voices() -> dict:
    """Return available voices and their language labels."""
    voices = list_available_voices()
    data: List[dict] = []
    for voice in voices:
        lang_code = get_language_code_from_voice(voice)
        data.append(
            {
                "name": voice,
                "language": LANGUAGE_CODES.get(lang_code, "Unknown"),
            }
        )

    return {"count": len(data), "voices": data}


@app.post("/synthesize")
def synthesize(request: SynthesisRequest):
    """Generate speech from text using a specified Kokoro voice."""
    normalized_voice = _standardize_voice_name(request.voice)
    available = list_available_voices()
    if normalized_voice not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{normalized_voice}' was not found. Call /voices for a full list.",
        )

    try:
        model = _get_pipeline(normalized_voice, request.device)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    lang_code = get_language_code_from_voice(normalized_voice)
    audio_tensor, phonemes = generate_speech(
        model=model,
        text=request.text,
        voice=normalized_voice,
        lang=lang_code,
        device=request.device,
        speed=request.speed,
    )

    if audio_tensor is None:
        raise HTTPException(status_code=500, detail="TTS pipeline returned no audio")

    if isinstance(audio_tensor, torch.Tensor):
        audio_np = audio_tensor.detach().cpu().numpy()
    elif isinstance(audio_tensor, np.ndarray):
        audio_np = audio_tensor
    else:
        raise HTTPException(status_code=500, detail="Unexpected audio type from pipeline")

    buffer = io.BytesIO()
    sf.write(buffer, audio_np, SAMPLE_RATE, format="WAV")
    buffer.seek(0)

    headers = {}
    if request.return_phonemes and phonemes:
        headers["X-Phonemes"] = phonemes

    return StreamingResponse(buffer, media_type="audio/wav", headers=headers)


def main() -> None:
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
