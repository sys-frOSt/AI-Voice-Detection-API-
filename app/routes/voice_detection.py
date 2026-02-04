"""
Voice Detection API Routes
Handles the main voice detection endpoint
"""
from typing import Optional
from pydantic import BaseModel, Field, field_validator

from fastapi import APIRouter, Request, Depends, HTTPException

from app.middleware.auth import verify_api_key
from app.audio.audio_processor import audio_processor
from app.voice_detector import voice_detector


# Create router
router = APIRouter(prefix="/api", tags=["Voice Detection"])


# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]


class VoiceDetectionRequest(BaseModel):
    """Request body for voice detection endpoint."""
    
    language: str = Field(
        ..., 
        description="Language of the audio. Must be one of: Tamil, English, Hindi, Malayalam, Telugu"
    )
    audioFormat: str = Field(
        ..., 
        description="Format of the audio. Must be 'mp3'"
    )
    audioBase64: str = Field(
        ..., 
        description="Base64-encoded MP3 audio data"
    )
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language must be one of: {', '.join(SUPPORTED_LANGUAGES)}")
        return v
    
    @field_validator('audioFormat')
    @classmethod
    def validate_format(cls, v):
        if v.lower() != 'mp3':
            raise ValueError("audioFormat must be 'mp3'")
        return v.lower()
    
    @field_validator('audioBase64')
    @classmethod
    def validate_audio(cls, v):
        if not v or len(v) < 100:
            raise ValueError("audioBase64 is required and must contain valid Base64 audio data")
        return v


class VoiceDetectionResponse(BaseModel):
    """Response body for successful voice detection."""
    
    status: str = Field(default="success", description="Response status")
    language: str = Field(..., description="Language of the analyzed audio")
    classification: str = Field(
        ..., 
        description="Classification result: 'AI_GENERATED' or 'HUMAN'"
    )
    confidenceScore: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score between 0.0 and 1.0"
    )
    explanation: str = Field(
        ..., 
        description="Short explanation for the classification decision"
    )


class ErrorResponse(BaseModel):
    """Response body for errors."""
    
    status: str = Field(default="error", description="Response status")
    message: str = Field(..., description="Error message")


@router.post(
    "/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        200: {"model": VoiceDetectionResponse, "description": "Successful detection"},
        400: {"model": ErrorResponse, "description": "Bad request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect AI-Generated Voice",
    description="Analyzes an MP3 audio sample and classifies it as AI-generated or human-spoken."
)
async def detect_voice(
    request: VoiceDetectionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Detect whether a voice sample is AI-generated or human-spoken.
    
    This endpoint accepts a Base64-encoded MP3 audio file and returns
    the classification result along with a confidence score.
    
    **Supported Languages:** Tamil, English, Hindi, Malayalam, Telugu
    """
    try:
        # Decode audio bytes for HF API
        import base64
        audio_bytes = base64.b64decode(request.audioBase64)
        
        # Process audio and extract features + raw samples
        features, audio_samples, sample_rate = audio_processor.process_audio_with_samples(
            request.audioBase64
        )
        
        # Detect voice type using ensemble (heuristic + local ML + HF API)
        result = voice_detector.detect(
            features, 
            audio=audio_samples, 
            sr=sample_rate,
            audio_bytes=audio_bytes
        )
        
        # Build response
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=result['classification'],
            confidenceScore=result['confidenceScore'],
            explanation=result['explanation']
        )
        
    except ValueError as e:
        # Handle audio processing errors
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": f"Audio processing error: {str(e)}"
            }
        )
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": f"Internal server error: {str(e)}"
            }
        )
