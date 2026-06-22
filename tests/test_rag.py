"""Unit tests for src/rag.py."""

import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from langchain_core.documents import Document


class TestFAQRetrieverInit:
    """Tests for FAQRetriever initialization and path logic."""

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_general_category_sets_general_paths(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        assert retriever.category == "general"
        assert "general_issues_faq.pdf" in retriever.pdf_path
        assert "vectorstore_general" in retriever.database_path

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_billing_category_sets_billing_paths(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="billing")

        assert retriever.category == "billing"
        assert "billing_faq.pdf" in retriever.pdf_path
        assert "vectorstore_billing" in retriever.database_path

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_category_is_lowered(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="BILLING")

        assert retriever.category == "billing"

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_default_category_is_general(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever()

        assert retriever.category == "general"


class TestFAQRetrieverLoadOrCreate:
    """Tests for load_or_create_database method."""

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.Chroma")
    @patch("os.path.exists", return_value=True)
    @patch("os.listdir", return_value=["some_file"])
    def test_loads_existing_database(self, mock_listdir, mock_exists, mock_chroma, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        mock_chroma.assert_called_once()
        assert retriever.vectorstore is not None

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.create_database_from_pdf")
    @patch("os.path.exists", return_value=False)
    def test_creates_database_when_not_exists(self, mock_exists, mock_create, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        mock_create.assert_called_once()


class TestFAQRetrieverSearch:
    """Tests for the search method."""

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_search_calls_similarity_search(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        mock_vectorstore = MagicMock()
        expected_docs = [Document(page_content="Result 1"), Document(page_content="Result 2")]
        mock_vectorstore.similarity_search.return_value = expected_docs
        retriever.vectorstore = mock_vectorstore

        results = retriever.search("test question", num_results=2)

        mock_vectorstore.similarity_search.assert_called_once_with("test question", k=2)
        assert results == expected_docs

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_search_default_num_results(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        mock_vectorstore = MagicMock()
        mock_vectorstore.similarity_search.return_value = []
        retriever.vectorstore = mock_vectorstore

        retriever.search("test question")

        mock_vectorstore.similarity_search.assert_called_once_with("test question", k=3)


class TestFAQRetrieverFormatResults:
    """Tests for the format_results method."""

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_format_empty_results(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        result = retriever.format_results([])
        assert result == "No relevant information found in FAQ."

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_format_single_result(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        docs = [Document(page_content="Answer about passwords")]
        result = retriever.format_results(docs)

        assert "[FAQ Section 1]" in result
        assert "Answer about passwords" in result

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_format_multiple_results(self, mock_load, mock_embed):
        from src.rag import FAQRetriever
        retriever = FAQRetriever(category="general")

        docs = [
            Document(page_content="First answer"),
            Document(page_content="Second answer"),
            Document(page_content="Third answer"),
        ]
        result = retriever.format_results(docs)

        assert "[FAQ Section 1]" in result
        assert "[FAQ Section 2]" in result
        assert "[FAQ Section 3]" in result
        assert "First answer" in result
        assert "Second answer" in result
        assert "Third answer" in result


class TestGetFAQRetriever:
    """Tests for the get_faq_retriever caching function."""

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_caching_returns_same_instance(self, mock_load, mock_embed):
        from src.rag import get_faq_retriever, _cached_retrievers
        _cached_retrievers.clear()

        r1 = get_faq_retriever("general")
        r2 = get_faq_retriever("general")

        assert r1 is r2

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_different_categories_return_different_instances(self, mock_load, mock_embed):
        from src.rag import get_faq_retriever, _cached_retrievers
        _cached_retrievers.clear()

        r1 = get_faq_retriever("general")
        r2 = get_faq_retriever("billing")

        assert r1 is not r2

    @patch("src.rag.HuggingFaceEmbeddings")
    @patch("src.rag.FAQRetriever.load_or_create_database")
    def test_category_is_lowered_for_cache(self, mock_load, mock_embed):
        from src.rag import get_faq_retriever, _cached_retrievers
        _cached_retrievers.clear()

        r1 = get_faq_retriever("General")
        r2 = get_faq_retriever("general")

        assert r1 is r2
