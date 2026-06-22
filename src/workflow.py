"""Workflow for the customer support system."""

import logging

from langgraph.graph import END, StateGraph

from src.agents import (
    analyze_sentiment,
    categorize,
    handle_billing,
    handle_general,
)
from src.router import route_query
from src.state import State

logger = logging.getLogger(__name__)


def create_workflow() -> StateGraph:
    """Create and configure the customer support workflow graph.

    Returns:
        Compiled LangGraph application.

    Raises:
        RuntimeError: If the workflow graph cannot be compiled.
    """
    try:
        workflow = StateGraph(State)

        workflow.add_node("categorize", categorize)
        workflow.add_node("analyze_sentiment", analyze_sentiment)
        workflow.add_node("handle_billing", handle_billing)
        workflow.add_node("handle_general", handle_general)

        workflow.add_edge("categorize", "analyze_sentiment")
        workflow.add_conditional_edges(
            "analyze_sentiment",
            route_query,
            {
                "handle_billing": "handle_billing",
                "handle_general": "handle_general",
            },
        )

        workflow.add_edge("handle_billing", END)
        workflow.add_edge("handle_general", END)

        workflow.set_entry_point("categorize")

        return workflow.compile()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to compile customer support workflow: {exc}"
        ) from exc


app = create_workflow()
