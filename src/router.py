"""Routing logic for the customer support workflow."""

import logging

from src.state import State

logger = logging.getLogger(__name__)


def route_query(state: State) -> str:
    """Route the query to the appropriate handler based on category.

    Returns:
        Name of the next node to execute.

    Raises:
        RuntimeError: If the category is missing from state.
    """
    category = state.get("category")
    if not category:
        raise RuntimeError(
            "Cannot route query: category was not set by the categorize step. "
            "The LLM may have failed to classify the query."
        )

    category = category.strip().lower()
    if "billing" in category:
        return "handle_billing"
    return "handle_general"
