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
    
    # Groq API Settings
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
    
    # Categories
    CATEGORIES = ["Technical", "Billing", "General"]
    
    # Sentiments
    SENTIMENTS = ["Positive", "Negative", "Neutral"]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required settings are present."""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in environment variables")
        return True

settings = Settings()
