"""LLM client setup."""

import logging

from langchain_groq import ChatGroq

from config.settings import settings

logger = logging.getLogger(__name__)


def get_llm() -> ChatGroq:
    """Initialize and return the Groq LLM client.

    Raises:
        ValueError: If GROQ_API_KEY is missing.
        ConnectionError: If the LLM client cannot be instantiated.
    """
    settings.validate()

    try:
        return ChatGroq(
            temperature=settings.TEMPERATURE,
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.MODEL_NAME,
        )
    except Exception as exc:
        raise ConnectionError(
            f"Failed to initialize LLM client (model={settings.MODEL_NAME}): {exc}"
        ) from exc


llm = get_llm()
