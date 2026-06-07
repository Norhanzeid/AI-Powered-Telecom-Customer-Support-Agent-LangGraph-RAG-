"""Tests for the FAQ Retrieval System."""
 
from src.rag import FAQRetriever, get_faq_retriever
 
 
def test_general_retriever():
    print("=" * 50)
    print("Testing GENERAL FAQ Retriever")
    print("=" * 50)
 
    retriever = FAQRetriever(category="general")
 
    questions = [
        "How do I reset my password?",
        "How can I track my order?",
        "How do I contact customer support?",
    ]
 
    for question in questions:
        print(f"\nQuestion: {question}")
        docs = retriever.search(question, num_results=2)
        context = retriever.format_results(docs)
        print(f"Retrieved {len(docs)} chunks")
        print(f"Context preview: {context[:200]}...")
 
 
def test_billing_retriever():
    print("\n" + "=" * 50)
    print("Testing BILLING FAQ Retriever")
    print("=" * 50)
 
    retriever = FAQRetriever(category="billing")
 
    questions = [
        "Why was my payment declined?",
        "How do I get a refund?",
        "What payment methods do you accept?",
    ]
 
    for question in questions:
        print(f"\nQuestion: {question}")
        docs = retriever.search(question, num_results=2)
        context = retriever.format_results(docs)
        print(f"Retrieved {len(docs)} chunks")
        print(f"Context preview: {context[:200]}...")
 
 
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