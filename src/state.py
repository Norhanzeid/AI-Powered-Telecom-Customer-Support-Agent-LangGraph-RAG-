"""State definition for the customer support workflow."""

from typing import TypedDict


class State(TypedDict):
    """State structure for the customer support workflow."""
    query: str
    category: str
    sentiment: str
    response: str
