"""Unit tests for src/utils.py."""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO


class TestRunCustomerSupport:
    """Tests for the run_customer_support function."""

    @patch("src.utils.app")
    def test_returns_expected_keys(self, mock_app):
        mock_app.invoke.return_value = {
            "query": "test query",
            "category": "General",
            "sentiment": "Neutral",
            "response": "Test response",
        }

        from src.utils import run_customer_support
        result = run_customer_support("test query")

        assert "category" in result
        assert "sentiment" in result
        assert "response" in result

    @patch("src.utils.app")
    def test_returns_correct_values(self, mock_app):
        mock_app.invoke.return_value = {
            "query": "billing question",
            "category": "Billing",
            "sentiment": "Negative",
            "response": "Billing answer",
        }

        from src.utils import run_customer_support
        result = run_customer_support("billing question")

        assert result["category"] == "Billing"
        assert result["sentiment"] == "Negative"
        assert result["response"] == "Billing answer"

    @patch("src.utils.app")
    def test_invokes_app_with_query(self, mock_app):
        mock_app.invoke.return_value = {
            "query": "my query",
            "category": "Technical",
            "sentiment": "Neutral",
            "response": "Tech response",
        }

        from src.utils import run_customer_support
        run_customer_support("my query")

        mock_app.invoke.assert_called_once_with({"query": "my query"})

    @patch("src.utils.app")
    def test_does_not_include_query_in_result(self, mock_app):
        mock_app.invoke.return_value = {
            "query": "test",
            "category": "General",
            "sentiment": "Neutral",
            "response": "Resp",
        }

        from src.utils import run_customer_support
        result = run_customer_support("test")

        assert "query" not in result


class TestPrintResult:
    """Tests for the print_result function."""

    @patch("src.utils.app")
    def test_print_result_outputs_query(self, mock_app, capsys):
        from src.utils import print_result
        result = {"category": "General", "sentiment": "Neutral", "response": "Response text"}
        print_result("My question", result)

        captured = capsys.readouterr()
        assert "My question" in captured.out

    @patch("src.utils.app")
    def test_print_result_outputs_category(self, mock_app, capsys):
        from src.utils import print_result
        result = {"category": "Billing", "sentiment": "Positive", "response": "Answer"}
        print_result("Q", result)

        captured = capsys.readouterr()
        assert "Billing" in captured.out

    @patch("src.utils.app")
    def test_print_result_outputs_sentiment(self, mock_app, capsys):
        from src.utils import print_result
        result = {"category": "General", "sentiment": "Negative", "response": "Answer"}
        print_result("Q", result)

        captured = capsys.readouterr()
        assert "Negative" in captured.out

    @patch("src.utils.app")
    def test_print_result_outputs_response(self, mock_app, capsys):
        from src.utils import print_result
        result = {"category": "General", "sentiment": "Neutral", "response": "Detailed response text"}
        print_result("Q", result)

        captured = capsys.readouterr()
        assert "Detailed response text" in captured.out
