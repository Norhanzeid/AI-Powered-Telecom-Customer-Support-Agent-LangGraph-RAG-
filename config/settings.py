"""Configuration settings for the Customer Support System."""

import logging
import os

import truststore
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Fix SSL certificate verification on Windows by using the OS certificate store
truststore.inject_into_ssl()

# Load environment variables from .env file
load_dotenv()


def _parse_temperature(raw: str) -> float:
    try:
        value = float(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"TEMPERATURE environment variable must be a number, got {raw!r}"
        ) from exc
    if not 0.0 <= value <= 2.0:
        raise ValueError(
            f"TEMPERATURE must be between 0.0 and 2.0, got {value}"
        )
    return value


class Settings:
    """Application settings."""

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    TEMPERATURE: float = _parse_temperature(os.getenv("TEMPERATURE", "0"))

    CATEGORIES = ["Technical", "Billing", "General"]
    SENTIMENTS = ["Positive", "Negative", "Neutral"]

    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or set the environment variable."
            )
        return True


settings = Settings()
