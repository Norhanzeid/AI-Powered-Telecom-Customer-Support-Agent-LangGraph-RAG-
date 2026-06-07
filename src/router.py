"""Routing logic for the customer support workflow."""

from src.state import State


def route_query(state: State) -> str:
    """
    Route the query based on category.
    All queries get FAQ answers, regardless of sentiment.
    
    Args:
        state: Current state containing sentiment and category
        
    Returns:
        Name of the next node to execute
    """
    category = state["category"].strip()
    
    # Route based on category (FAQ will answer regardless of sentiment)
    if "technical" in category.lower():
        return "handle_technical"
    elif "billing" in category.lower():
        return "handle_billing"
    else:
        return "handle_general"
