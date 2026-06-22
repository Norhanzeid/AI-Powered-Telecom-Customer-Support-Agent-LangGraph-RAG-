"""Configuration settings for the Customer Support System."""

import os

import truststore
from dotenv import load_dotenv

# Fix SSL certificate verification on Windows by using the OS certificate store
truststore.inject_into_ssl()

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings."""

    # Categories
    CATEGORIES = ["Technical", "Billing", "General"]

    # Sentiments
    SENTIMENTS = ["Positive", "Negative", "Neutral"]

    @property
    def GROQ_API_KEY(self) -> str:
        """Retrieve API key from environment at access time (never stored as a class attribute)."""
        return os.getenv("GROQ_API_KEY", "")

    @property
    def MODEL_NAME(self) -> str:
        return os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")

    @property
    def TEMPERATURE(self) -> float:
        return float(os.getenv("TEMPERATURE", "0"))

    def validate(self) -> bool:
        """Validate that required settings are present."""
        if not self.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Please add it to your .env file or set it as an environment variable."
            )
        return True


settings = Settings()
