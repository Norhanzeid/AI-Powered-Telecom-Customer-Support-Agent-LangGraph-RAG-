"""LLM client setup."""
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from config.settings import settings

def get_llm() -> ChatOpenAI:
    """
    Initialize and return the OpenAI LLM client.
    
    Returns:
        ChatOpenAI: Configured LLM instance
    """
    settings.validate()
    
    return ChatOpenAI(
        temperature=settings.TEMPERATURE,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        model=settings.MODEL_NAME
    )
# Initialize the LLM instance
llm = get_llm()
