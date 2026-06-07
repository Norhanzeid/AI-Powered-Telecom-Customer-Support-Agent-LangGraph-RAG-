"""Main application entry point for the Customer Support System."""

import sys
from src.utils import run_customer_support, print_result


def main():
    """Run the customer support system with example queries."""
    
    # Example queries
    queries = [
        "I can't access my account",
        "I forget my password",
        "Where can I find my receipt",
        "This is terrible! I'm very frustrated!"
    ]
    
    print("=" * 70)
    print("Customer Support System - Running Example Queries")
    print("=" * 70)
    print()
    
    for query in queries:
        try:
            result = run_customer_support(query)
            print_result(query, result)
        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            print()
    
    print("=" * 70)

if __name__ == "__main__":
    main()
