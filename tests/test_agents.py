"""Unit tests for src/agents.py."""

import pytest
from unittest.mock import patch, MagicMock


class TestCategorize:
    """Tests for the categorize agent function."""

    @patch("src.agents.llm")
    def test_categorize_returns_category(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "Billing"
        mock_llm.__or__ = MagicMock()
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import categorize
            state = {"query": "Why was my payment declined?", "category": "", "sentiment": "", "response": ""}
            result = categorize(state)

            assert result["category"] == "Billing"

    @patch("src.agents.llm")
    def test_categorize_strips_whitespace(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "  Technical  "
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import categorize
            state = {"query": "Website is not loading", "category": "", "sentiment": "", "response": ""}
            result = categorize(state)

            assert result["category"] == "Technical"

    @patch("src.agents.llm")
    def test_categorize_general(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "General"
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import categorize
            state = {"query": "How do I reset my password?", "category": "", "sentiment": "", "response": ""}
            result = categorize(state)

            assert result["category"] == "General"


class TestAnalyzeSentiment:
    """Tests for the analyze_sentiment agent function."""

    @patch("src.agents.llm")
    def test_analyze_sentiment_negative(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "Negative"
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import analyze_sentiment
            state = {"query": "This is terrible!", "category": "", "sentiment": "", "response": ""}
            result = analyze_sentiment(state)

            assert result["sentiment"] == "Negative"

    @patch("src.agents.llm")
    def test_analyze_sentiment_positive(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "Positive"
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import analyze_sentiment
            state = {"query": "Great service!", "category": "", "sentiment": "", "response": ""}
            result = analyze_sentiment(state)

            assert result["sentiment"] == "Positive"

    @patch("src.agents.llm")
    def test_analyze_sentiment_neutral(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "Neutral"
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import analyze_sentiment
            state = {"query": "What are your hours?", "category": "", "sentiment": "", "response": ""}
            result = analyze_sentiment(state)

            assert result["sentiment"] == "Neutral"

    @patch("src.agents.llm")
    def test_analyze_sentiment_strips_whitespace(self, mock_llm):
        mock_response = MagicMock()
        mock_response.content = "  Positive  "
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import analyze_sentiment
            state = {"query": "Thank you!", "category": "", "sentiment": "", "response": ""}
            result = analyze_sentiment(state)

            assert result["sentiment"] == "Positive"


class TestHandleBilling:
    """Tests for the handle_billing agent function."""

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_billing_returns_response(self, mock_llm, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [MagicMock(page_content="Billing FAQ content")]
        mock_retriever.format_results.return_value = "[FAQ Section 1]\nBilling FAQ content"
        mock_get_retriever.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "Here is your billing answer."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_billing
            state = {"query": "Why was I charged twice?", "category": "Billing", "sentiment": "Neutral", "response": ""}
            result = handle_billing(state)

            assert result["response"] == "Here is your billing answer."
            mock_get_retriever.assert_called_with(category="billing")

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_billing_negative_sentiment_adds_escalation(self, mock_llm, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever.format_results.return_value = "No relevant information found in FAQ."
        mock_get_retriever.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "Billing response."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_billing
            state = {"query": "I'm furious about charges!", "category": "Billing", "sentiment": "Negative", "response": ""}
            result = handle_billing(state)

            assert "human agent will follow up" in result["response"]

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_billing_neutral_no_escalation(self, mock_llm, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever.format_results.return_value = "No relevant information found in FAQ."
        mock_get_retriever.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "Here is the billing info."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_billing
            state = {"query": "What are your billing hours?", "category": "Billing", "sentiment": "Neutral", "response": ""}
            result = handle_billing(state)

            assert "human agent will follow up" not in result["response"]

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_billing_fallback_on_rag_error(self, mock_llm, mock_get_retriever):
        mock_get_retriever.side_effect = Exception("RAG error")

        mock_response = MagicMock()
        mock_response.content = "Fallback billing response."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_billing
            state = {"query": "Billing question", "category": "Billing", "sentiment": "Neutral", "response": ""}
            result = handle_billing(state)

            assert result["response"] == "Fallback billing response."


class TestHandleGeneral:
    """Tests for the handle_general agent function."""

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_general_returns_response(self, mock_llm, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [MagicMock(page_content="General FAQ content")]
        mock_retriever.format_results.return_value = "[FAQ Section 1]\nGeneral FAQ content"
        mock_get_retriever.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "Here is your general answer."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_general
            state = {"query": "How do I reset my password?", "category": "General", "sentiment": "Neutral", "response": ""}
            result = handle_general(state)

            assert result["response"] == "Here is your general answer."
            mock_get_retriever.assert_called_with(category="general")

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_general_negative_sentiment_adds_escalation(self, mock_llm, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever.format_results.return_value = "No relevant information found in FAQ."
        mock_get_retriever.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "General response."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_general
            state = {"query": "This is terrible!", "category": "General", "sentiment": "Negative", "response": ""}
            result = handle_general(state)

            assert "human agent will follow up" in result["response"]

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_general_positive_no_escalation(self, mock_llm, mock_get_retriever):
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = []
        mock_retriever.format_results.return_value = "No relevant information found in FAQ."
        mock_get_retriever.return_value = mock_retriever

        mock_response = MagicMock()
        mock_response.content = "Glad you like our service!"
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_general
            state = {"query": "Great service!", "category": "General", "sentiment": "Positive", "response": ""}
            result = handle_general(state)

            assert "human agent will follow up" not in result["response"]

    @patch("src.agents.get_faq_retriever")
    @patch("src.agents.llm")
    def test_handle_general_fallback_on_rag_error(self, mock_llm, mock_get_retriever):
        mock_get_retriever.side_effect = Exception("RAG error")

        mock_response = MagicMock()
        mock_response.content = "Fallback general response."
        chain_mock = MagicMock()
        chain_mock.invoke.return_value = mock_response

        with patch("src.agents.ChatPromptTemplate") as mock_prompt_cls:
            mock_prompt = MagicMock()
            mock_prompt_cls.from_template.return_value = mock_prompt
            mock_prompt.__or__ = MagicMock(return_value=chain_mock)

            from src.agents import handle_general
            state = {"query": "General question", "category": "General", "sentiment": "Neutral", "response": ""}
            result = handle_general(state)

            assert result["response"] == "Fallback general response."
