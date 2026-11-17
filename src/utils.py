from typing import Dict
from src.workflow import app

def run_customer_support(query: str) -> Dict[str, str]:
    """
    Process a customer support query through the workflow.
    
    Args:
        query: The customer's question or issue
        
    Returns:
        Dictionary containing category, sentiment, and response
    """
    results = app.invoke({"query": query})
    return {
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }


def print_result(query: str, result: Dict[str, str]) -> None:
    """
    Print the query and results in a formatted manner.
    
    Args:
        query: The original query
        result: Dictionary containing category, sentiment, and response
    """
    print(f"Query: {query}")
    print(f"Category: {result['category']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Response: {result['response']}")
    print("\n")
