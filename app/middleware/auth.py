"""
Authentication Middleware
Handles API key validation for the voice detection API
"""
import os
from typing import Optional

from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key configuration
API_KEY_NAME = "x-api-key"
API_KEY = os.getenv("API_KEY", "add_your_default_api_key_here")

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(request: Request) -> str:
    """
    Verify the API key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    api_key = request.headers.get(API_KEY_NAME)

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Missing API key. Please provide x-api-key header.",
            },
        )

    if api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Invalid API key or malformed request",
            },
        )

    return api_key


def get_api_key() -> str:
    """Get the configured API key (for documentation purposes)."""
    return API_KEY
