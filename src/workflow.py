"""Workflow for the customer support system."""

from langgraph.graph import StateGraph, END
from src.state import State
from src.agents import (
    categorize,
    analyze_sentiment,
    handle_technical,
    handle_billing,
    handle_general
)
from src.router import route_query


def create_workflow() -> StateGraph:
    """
    Create and configure the customer support workflow graph.
    
    Returns:
        Compiled workflow application
    """
    workflow = StateGraph(State)
     
    # Add nodes
    workflow.add_node("categorize", categorize)
    workflow.add_node("analyze_sentiment", analyze_sentiment)
    workflow.add_node("handle_technical", handle_technical)
    workflow.add_node("handle_billing", handle_billing)
    workflow.add_node("handle_general", handle_general)
    
    # Add edges
    workflow.add_edge("categorize", "analyze_sentiment")
    workflow.add_conditional_edges(
        "analyze_sentiment",
        route_query,
        {
            "handle_technical": "handle_technical",
            "handle_billing": "handle_billing",
            "handle_general": "handle_general"
        }
    )
    
    # Add terminal edges
    workflow.add_edge("handle_technical", END)
    workflow.add_edge("handle_billing", END)
    workflow.add_edge("handle_general", END)
    
    # Set entry point
    workflow.set_entry_point("categorize")
    
    return workflow.compile()


# Create the compiled workflow application
app = create_workflow()
