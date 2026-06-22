"""Unit tests for src/state.py."""

import pytest
from src.state import State


class TestState:
    """Tests for the State TypedDict."""

    def test_state_accepts_valid_keys(self):
        state: State = {
            "query": "test query",
            "category": "General",
            "sentiment": "Neutral",
            "response": "Some response",
        }
        assert state["query"] == "test query"
        assert state["category"] == "General"
        assert state["sentiment"] == "Neutral"
        assert state["response"] == "Some response"

    def test_state_has_expected_annotations(self):
        annotations = State.__annotations__
        assert "query" in annotations
        assert "category" in annotations
        assert "sentiment" in annotations
        assert "response" in annotations
        assert annotations["query"] is str
        assert annotations["category"] is str
        assert annotations["sentiment"] is str
        assert annotations["response"] is str

    def test_state_has_four_fields(self):
        assert len(State.__annotations__) == 4
