"""Unit tests for src/router.py."""

import pytest
from src.router import route_query


class TestRouteQuery:
    """Tests for the route_query function."""

    def test_billing_category_routes_to_handle_billing(self):
        state = {"query": "test", "category": "Billing", "sentiment": "Neutral"}
        assert route_query(state) == "handle_billing"

    def test_billing_lowercase_routes_to_handle_billing(self):
        state = {"query": "test", "category": "billing", "sentiment": "Neutral"}
        assert route_query(state) == "handle_billing"

    def test_billing_with_whitespace_routes_to_handle_billing(self):
        state = {"query": "test", "category": "  Billing  ", "sentiment": "Neutral"}
        assert route_query(state) == "handle_billing"

    def test_billing_mixed_case_routes_to_handle_billing(self):
        state = {"query": "test", "category": "BILLING", "sentiment": "Neutral"}
        assert route_query(state) == "handle_billing"

    def test_general_category_routes_to_handle_general(self):
        state = {"query": "test", "category": "General", "sentiment": "Neutral"}
        assert route_query(state) == "handle_general"

    def test_technical_category_routes_to_handle_general(self):
        state = {"query": "test", "category": "Technical", "sentiment": "Neutral"}
        assert route_query(state) == "handle_general"

    def test_unknown_category_routes_to_handle_general(self):
        state = {"query": "test", "category": "Unknown", "sentiment": "Neutral"}
        assert route_query(state) == "handle_general"

    def test_empty_category_routes_to_handle_general(self):
        state = {"query": "test", "category": "", "sentiment": "Neutral"}
        assert route_query(state) == "handle_general"

    def test_negative_sentiment_billing_still_routes_to_billing(self):
        state = {"query": "test", "category": "Billing", "sentiment": "Negative"}
        assert route_query(state) == "handle_billing"

    def test_negative_sentiment_general_still_routes_to_general(self):
        state = {"query": "test", "category": "General", "sentiment": "Negative"}
        assert route_query(state) == "handle_general"

    def test_positive_sentiment_does_not_affect_routing(self):
        state = {"query": "test", "category": "Billing", "sentiment": "Positive"}
        assert route_query(state) == "handle_billing"
