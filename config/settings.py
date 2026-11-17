"""Configuration settings for the Customer Support System."""

import os
import truststore
from dotenv import load_dotenv

# Fix SSL certificate verification on Windows by using the OS certificate store
truststore.inject_into_ssl()

# Load environment variables from .env file
load_dotenv()
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


class Settings:
    """Application settings."""
    
    # OpenAI API Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("LLM_MODEL", "gpt-4o")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # Categories
    CATEGORIES = ["Technical", "Billing", "General"]
    
    # Sentiments
    SENTIMENTS = ["Positive", "Negative", "Neutral"]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in environment variables")
        return True

settings = Settings()
