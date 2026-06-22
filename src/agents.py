"""Agent nodes for the customer support workflow."""

from src.state import State
from src.shared import classify_query, handle_rag_query


# RAG handler configuration per category
_RAG_CONFIG = {
    "billing": {
        "category": "billing",
        "role": "billing support assistant",
        "db_label": "Billing FAQ",
        "fallback_label": "billing",
    },
    "general": {
        "category": "general",
        "role": "customer support assistant",
        "db_label": "General FAQ",
        "fallback_label": "general",
    },
}


def categorize(state: State) -> State:
    """
    Categorize the customer query into Technical, Billing, or General.

    Args:
        state: Current state containing the query

    Returns:
        Updated state with category
    """
    prompt_text = (
        "Categorize the following customer query into one of these categories:\n\n"
        "- **Technical**: Website errors, technical bugs, system failures, app crashes, connectivity issues\n"
        "- **Billing**: Payments, invoices, receipts, refunds, charges, promo codes, subscriptions, pricing, fees\n"
        "- **General**: Account access (password reset, username recovery), profile settings, orders, "
        "shipping, returns, customer service contact info, mobile apps, browser support, policies, FAQs\n\n"
        "IMPORTANT: Respond with ONLY ONE WORD - either 'Technical', 'Billing', or 'General'.\n"
        "Do not provide any explanation or additional text.\n\n"
        "Examples:\n"
        "- 'How do I reset my password?' \u2192 General\n"
        "- 'Why was my payment declined?' \u2192 Billing\n"
        "- 'The website is not loading' \u2192 Technical\n\n"
        "Query: {query}\n\n"
        "Category:"
    )
    category = classify_query(prompt_text, state["query"])
    return {"category": category}


def analyze_sentiment(state: State) -> State:
    """
    Analyze the sentiment of the customer query.

    Args:
        state: Current state containing the query

    Returns:
        Updated state with sentiment (Positive, Negative, or Neutral)
    """
    prompt_text = (
        "Analyze the sentiment of the following customer query.\n\n"
        "IMPORTANT: Respond with ONLY ONE WORD - either 'Positive', 'Negative', or 'Neutral'. "
        "Do not provide any explanation or additional text.\n\n"
        "Query: {query}\n\n"
        "Sentiment:"
    )
    sentiment = classify_query(prompt_text, state["query"])
    return {"sentiment": sentiment}


def handle_billing(state: State) -> State:
    """
    Handle billing-related queries using RAG (Retrieval-Augmented Generation).

    Args:
        state: Current state containing the query

    Returns:
        Updated state with billing response
    """
    return handle_rag_query(state, _RAG_CONFIG["billing"])


def handle_general(state: State) -> State:
    """
    Handle general customer queries using RAG (Retrieval-Augmented Generation).

    Args:
        state: Current state containing the query

    Returns:
        Updated state with general response
    """
    return handle_rag_query(state, _RAG_CONFIG["general"])
