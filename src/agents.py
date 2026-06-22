"""Agent nodes for the customer support workflow."""

import logging

from langchain_core.prompts import ChatPromptTemplate
from src.state import State
from src.llm import llm
from src.rag import get_faq_retriever

logger = logging.getLogger(__name__)


def categorize(state: State) -> State:
    """
    Categorize the customer query into Technical, Billing, or General.
    
    Args:
        state: Current state containing the query
        
    Returns: 
        Updated state with category
    """
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories:\n\n"
        "- **Technical**: Website errors, technical bugs, system failures, app crashes, connectivity issues\n"
        "- **Billing**: Payments, invoices, receipts, refunds, charges, promo codes, subscriptions, pricing, fees\n"
        "- **General**: Account access (password reset, username recovery), profile settings, orders, "
        "shipping, returns, customer service contact info, mobile apps, browser support, policies, FAQs\n\n"
        "IMPORTANT: Respond with ONLY ONE WORD - either 'Technical', 'Billing', or 'General'.\n"
        "Do not provide any explanation or additional text.\n\n"
        "Examples:\n"
        "- 'How do I reset my password?' → General\n"
        "- 'Why was my payment declined?' → Billing\n"
        "- 'The website is not loading' → Technical\n\n"
        "Query: {query}\n\n"
        "Category:"
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"]}).content.strip()
    return {"category": category}


def analyze_sentiment(state: State) -> State:
    """
    Analyze the sentiment of the customer query.
    
    Args:
        state: Current state containing the query
        
    Returns:
        Updated state with sentiment (Positive, Negative, or Neutral)
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query.\n\n"
        "IMPORTANT: Respond with ONLY ONE WORD - either 'Positive', 'Negative', or 'Neutral'. "
        "Do not provide any explanation or additional text.\n\n"
        "Query: {query}\n\n"
        "Sentiment:"
    )
    chain = prompt | llm
    sentiment = chain.invoke({"query": state["query"]}).content.strip()
    return {"sentiment": sentiment}


def handle_billing(state: State) -> State:
    """
    Handle billing-related queries using RAG (Retrieval-Augmented Generation).
    Retrieves relevant information from the Billing FAQ PDF and generates a response.
    
    Args:
        state: Current state containing the query
        
    Returns:
        Updated state with billing response
    """
    query = state["query"]
    
    try:
        # Get billing FAQ retriever
        retriever = get_faq_retriever(category="billing")
        
        # Retrieve relevant documents
        docs = retriever.search(query, num_results=3)
        
        # Format the context
        context = retriever.format_results(docs)
        
        # Create prompt with context
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful billing support assistant. "
            "Use the following context from our Billing FAQ database to answer the customer's query. "
            "If the context doesn't contain the answer, provide a helpful billing-related response.\n\n"
            "Context:\n{context}\n\n"
            "Customer Query: {query}\n\n"
            "Response:"
        )
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": query}).content
        
    except Exception as e:
        logger.exception("Error in Billing RAG retrieval")
        prompt = ChatPromptTemplate.from_template(
            "Provide a billing response to the following customer query: {query}"
        )
        chain = prompt | llm
        response = chain.invoke({"query": query}).content
    
    # Add escalation note if sentiment is negative
    sentiment = state.get("sentiment", "").strip().lower()
    if "negative" in sentiment:
        response += "\n\n Note: Due to the urgency of your concern, a human agent will follow up with you shortly."
    
    return {"response": response}


def handle_general(state: State) -> State:
    """
    Handle general customer queries using RAG (Retrieval-Augmented Generation).
    Retrieves relevant information from the FAQ PDF and generates a response.
    
    Args:
        state: Current state containing the query
        
    Returns:
        Updated state with general response
    """
    query = state["query"]
    
    try:
        # Get general FAQ retriever
        retriever = get_faq_retriever(category="general")
        
        # Retrieve relevant documents
        docs = retriever.search(query, num_results=3)
        
        # Format the context
        context = retriever.format_results(docs)
        
        # Create prompt with context
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful customer support assistant. "
            "Use the following context from our General FAQ database to answer the customer's query. "
            "If the context doesn't contain the answer, provide a helpful general response.\n\n"
            "Context:\n{context}\n\n"
            "Customer Query: {query}\n\n"
            "Response:"
        )
        
        chain = prompt | llm
        response = chain.invoke({"context": context, "query": query}).content
        
    except Exception as e:
        logger.exception("Error in General RAG retrieval")
        prompt = ChatPromptTemplate.from_template(
            "Provide a general response to the following customer query: {query}"
        )
        chain = prompt | llm
        response = chain.invoke({"query": query}).content
    
    # Add escalation note if sentiment is negative
    sentiment = state.get("sentiment", "").strip().lower()
    if "negative" in sentiment:
        response += "\n\n Note: Due to the urgency of your concern, a human agent will follow up with you shortly."
    
    return {"response": response}
