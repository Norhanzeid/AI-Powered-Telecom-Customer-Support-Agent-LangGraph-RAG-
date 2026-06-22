"""Shared utilities for the customer support workflow.

This module consolidates duplicated patterns from agent nodes into
reusable helper functions for LLM classification and RAG-based response
generation.
"""

from typing import Dict

from langchain_core.prompts import ChatPromptTemplate

from src.llm import llm
from src.rag import get_faq_retriever
from src.state import State

ESCALATION_NOTE = (
    "\n\n Note: Due to the urgency of your concern, "
    "a human agent will follow up with you shortly."
)


def classify_query(prompt_text: str, query: str) -> str:
    """Invoke the LLM with a classification prompt and return the stripped label.

    Used by both the categorize and analyze_sentiment agent nodes to avoid
    repeating the prompt-creation / chain-invocation boilerplate.

    Args:
        prompt_text: A prompt template string containing a ``{query}`` placeholder.
        query: The customer query to classify.

    Returns:
        The LLM's single-word classification label (stripped of whitespace).
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm
    return chain.invoke({"query": query}).content.strip()


def handle_rag_query(state: State, config: Dict[str, str]) -> State:
    """Handle a customer query using RAG retrieval and LLM generation.

    Consolidates the duplicated logic from handle_billing and handle_general
    into a single configurable function.

    Args:
        state: Current workflow state containing the query and sentiment.
        config: Dictionary with keys:
            - category: retriever category ("billing" or "general")
            - role: assistant role for the system prompt
            - db_label: label for the FAQ database in the prompt
            - fallback_label: label used in the fallback prompt

    Returns:
        Updated state dict with the generated response.
    """
    query = state["query"]

    try:
        retriever = get_faq_retriever(category=config["category"])
        docs = retriever.search(query, num_results=3)
        context = retriever.format_results(docs)

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful {role}. "
            "Use the following context from our {db_label} database to answer the customer's query. "
            "If the context doesn't contain the answer, provide a helpful {fallback_label}-related response.\n\n"
            "Context:\n{context}\n\n"
            "Customer Query: {query}\n\n"
            "Response:"
        )
        chain = prompt | llm
        response = chain.invoke({
            "role": config["role"],
            "db_label": config["db_label"],
            "fallback_label": config["fallback_label"],
            "context": context,
            "query": query,
        }).content

    except Exception as e:
        print(f"Error in {config['category'].title()} RAG retrieval: {e}")
        prompt = ChatPromptTemplate.from_template(
            "Provide a {fallback_label} response to the following customer query: {query}"
        )
        chain = prompt | llm
        response = chain.invoke({
            "fallback_label": config["fallback_label"],
            "query": query,
        }).content

    sentiment = state.get("sentiment", "").strip().lower()
    if "negative" in sentiment:
        response += ESCALATION_NOTE

    return {"response": response}
