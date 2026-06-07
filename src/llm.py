"""LLM client setup."""
from langchain_groq import ChatGroq
from config.settings import settings

def get_llm() -> ChatGroq:
    """
    Initialize and return the Groq LLM client.
    
    Returns:
        ChatGroq: Configured LLM instance
    """
    settings.validate()
    
    return ChatGroq(
        temperature=settings.TEMPERATURE,
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME
    )
# Initialize the LLM instance
llm = get_llm()
