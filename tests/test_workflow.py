"""Unit tests for src/workflow.py."""

import pytest
from unittest.mock import patch, MagicMock


class TestCreateWorkflow:
    """Tests for the create_workflow function."""

    @patch("src.workflow.handle_general")
    @patch("src.workflow.handle_billing")
    @patch("src.workflow.analyze_sentiment")
    @patch("src.workflow.categorize")
    @patch("src.workflow.route_query")
    def test_workflow_compiles_successfully(
        self, mock_route, mock_cat, mock_sent, mock_billing, mock_general
    ):
        from src.workflow import create_workflow
        app = create_workflow()
        assert app is not None

    @patch("src.workflow.handle_general")
    @patch("src.workflow.handle_billing")
    @patch("src.workflow.analyze_sentiment")
    @patch("src.workflow.categorize")
    @patch("src.workflow.route_query")
    def test_workflow_has_invoke_method(
        self, mock_route, mock_cat, mock_sent, mock_billing, mock_general
    ):
        from src.workflow import create_workflow
        app = create_workflow()
        assert hasattr(app, "invoke")

    @patch("src.workflow.handle_general")
    @patch("src.workflow.handle_billing")
    @patch("src.workflow.analyze_sentiment")
    @patch("src.workflow.categorize")
    @patch("src.workflow.route_query")
    def test_workflow_routes_billing_query(
        self, mock_route, mock_cat, mock_sent, mock_billing, mock_general
    ):
        mock_cat.return_value = {"category": "Billing"}
        mock_sent.return_value = {"sentiment": "Neutral"}
        mock_route.return_value = "handle_billing"
        mock_billing.return_value = {"response": "Billing answer"}

        from src.workflow import create_workflow
        app = create_workflow()
        result = app.invoke({"query": "Why was I charged?"})

        assert result["category"] == "Billing"
        assert result["response"] == "Billing answer"

    @patch("src.workflow.handle_general")
    @patch("src.workflow.handle_billing")
    @patch("src.workflow.analyze_sentiment")
    @patch("src.workflow.categorize")
    @patch("src.workflow.route_query")
    def test_workflow_routes_general_query(
        self, mock_route, mock_cat, mock_sent, mock_billing, mock_general
    ):
        mock_cat.return_value = {"category": "General"}
        mock_sent.return_value = {"sentiment": "Positive"}
        mock_route.return_value = "handle_general"
        mock_general.return_value = {"response": "General answer"}

        from src.workflow import create_workflow
        app = create_workflow()
        result = app.invoke({"query": "How do I reset my password?"})

        assert result["category"] == "General"
        assert result["response"] == "General answer"

    @patch("src.workflow.handle_general")
    @patch("src.workflow.handle_billing")
    @patch("src.workflow.analyze_sentiment")
    @patch("src.workflow.categorize")
    @patch("src.workflow.route_query")
    def test_workflow_preserves_sentiment(
        self, mock_route, mock_cat, mock_sent, mock_billing, mock_general
    ):
        mock_cat.return_value = {"category": "General"}
        mock_sent.return_value = {"sentiment": "Negative"}
        mock_route.return_value = "handle_general"
        mock_general.return_value = {"response": "Answer with escalation"}

        from src.workflow import create_workflow
        app = create_workflow()
        result = app.invoke({"query": "This is awful!"})

        assert result["sentiment"] == "Negative"
