"""Tests for the FAQ Retrieval System."""

from typing import List

from src.rag import FAQRetriever, get_faq_retriever


def _test_retriever(category: str, questions: List[str]) -> None:
    """Run a standard retrieval test for the given category and questions.

    Args:
        category: FAQ category to test ("general" or "billing").
        questions: Sample questions to query against the retriever.
    """
    label = category.upper()
    print("=" * 50)
    print(f"Testing {label} FAQ Retriever")
    print("=" * 50)

    retriever = FAQRetriever(category=category)

    for question in questions:
        print(f"\nQuestion: {question}")
        docs = retriever.search(question, num_results=2)
        context = retriever.format_results(docs)
        print(f"Retrieved {len(docs)} chunks")
        print(f"Context preview: {context[:200]}...")


def test_general_retriever():
    _test_retriever("general", [
        "How do I reset my password?",
        "How can I track my order?",
        "How do I contact customer support?",
    ])


def test_billing_retriever():
    print()
    _test_retriever("billing", [
        "Why was my payment declined?",
        "How do I get a refund?",
        "What payment methods do you accept?",
    ])


def test_cached_retriever():
    print("\n" + "=" * 50)
    print("Testing Cached Retriever")
    print("=" * 50)

    r1 = get_faq_retriever("general")
    r2 = get_faq_retriever("general")

    if r1 is r2:
        print("Caching works correctly - same instance returned.")
    else:
        print("WARNING: Caching not working - different instances returned.")


if __name__ == "__main__":
    test_general_retriever()
    test_billing_retriever()
    test_cached_retriever()
    print("\nAll tests completed!")