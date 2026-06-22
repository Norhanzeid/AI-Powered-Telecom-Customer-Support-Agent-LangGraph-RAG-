"""Utilities for running the customer support workflow."""

import logging
from typing import Dict

from src.workflow import app

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("category", "sentiment", "response")


def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer support query through the workflow.

    Args:
        query: The customer's question or issue.

    Returns:
        Dictionary with keys ``category``, ``sentiment``, and ``response``.

    Raises:
        ValueError: If the query is empty.
        RuntimeError: If the workflow fails or returns incomplete results.
    """
    if not query or not query.strip():
        raise ValueError("Query must not be empty.")

    try:
        results = app.invoke({"query": query})
    except Exception as exc:
        raise RuntimeError(
            f"Workflow failed while processing query: {exc}"
        ) from exc

    missing = [k for k in _REQUIRED_KEYS if not results.get(k)]
    if missing:
        raise RuntimeError(
            f"Workflow returned incomplete results (missing: {', '.join(missing)}). "
            f"One or more processing steps may have failed."
        )

    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"],
    }


def print_result(query: str, result: Dict[str, str]) -> None:
    """Print the query and results in a formatted manner."""
    print(f"Query: {query}")
    print(f"Category: {result['category']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Response: {result['response']}")
    print("\n")
