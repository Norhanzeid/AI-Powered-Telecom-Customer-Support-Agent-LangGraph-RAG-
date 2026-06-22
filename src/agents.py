"""Agent nodes for the customer support workflow."""

import logging

from langchain_core.prompts import ChatPromptTemplate

from src.llm import llm
from src.rag import get_faq_retriever
from src.state import State

logger = logging.getLogger(__name__)


def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General.

    Raises:
        RuntimeError: If the LLM fails to categorize the query.
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
    try:
        category = chain.invoke({"query": state["query"]}).content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to categorize query: {exc}"
        ) from exc
    return {"category": category}


def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query.

    Raises:
        RuntimeError: If the LLM fails to analyze sentiment.
    """
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query.\n\n"
        "IMPORTANT: Respond with ONLY ONE WORD - either 'Positive', 'Negative', or 'Neutral'. "
        "Do not provide any explanation or additional text.\n\n"
        "Query: {query}\n\n"
        "Sentiment:"
    )
    chain = prompt | llm
    try:
        sentiment = chain.invoke({"query": state["query"]}).content.strip()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to analyze sentiment: {exc}"
        ) from exc
    return {"sentiment": sentiment}


def _generate_rag_response(category: str, query: str, system_role: str) -> str:
    """Retrieve FAQ context and generate an LLM response.

    Returns the LLM response text.

    Raises:
        RuntimeError: If both RAG-augmented and plain LLM generation fail.
    """
    rag_error = None
    try:
        retriever = get_faq_retriever(category=category)
        docs = retriever.search(query, num_results=3)
        context = retriever.format_results(docs)

        prompt = ChatPromptTemplate.from_template(
            f"You are a helpful {system_role} support assistant. "
            "Use the following context from our FAQ database to answer the customer's query. "
            "If the context doesn't contain the answer, provide a helpful response.\n\n"
            "Context:\n{context}\n\n"
            "Customer Query: {query}\n\n"
            "Response:"
        )
        chain = prompt | llm
        return chain.invoke({"context": context, "query": query}).content
    except Exception as exc:
        rag_error = exc
        logger.warning(
            "RAG retrieval failed for %s query, falling back to plain LLM: %s",
            category, exc,
        )

    try:
        prompt = ChatPromptTemplate.from_template(
            f"Provide a {system_role} response to the following customer query: {{query}}"
        )
        chain = prompt | llm
        return chain.invoke({"query": query}).content
    except Exception as exc:
        raise RuntimeError(
            f"Both RAG and fallback LLM failed for {category} query. "
            f"RAG error: {rag_error}; LLM error: {exc}"
        ) from exc


def handle_billing(state: State) -> State:
    """Handle billing-related queries using RAG.

    Raises:
        RuntimeError: If both RAG and fallback LLM generation fail.
    """
    query = state["query"]
    response = _generate_rag_response("billing", query, "billing")

    sentiment = state.get("sentiment", "").strip().lower()
    if "negative" in sentiment:
        response += (
            "\n\n Note: Due to the urgency of your concern, "
            "a human agent will follow up with you shortly."
        )

    return {"response": response}


def handle_general(state: State) -> State:
    """Handle general customer queries using RAG.

    Raises:
        RuntimeError: If both RAG and fallback LLM generation fail.
    """
    query = state["query"]
    response = _generate_rag_response("general", query, "customer")

    sentiment = state.get("sentiment", "").strip().lower()
    if "negative" in sentiment:
        response += (
            "\n\n Note: Due to the urgency of your concern, "
            "a human agent will follow up with you shortly."
        )

    return {"response": response}
