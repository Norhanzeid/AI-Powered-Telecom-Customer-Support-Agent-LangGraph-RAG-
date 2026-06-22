"""Unit tests for config/settings.py."""

import os
import pytest
from unittest.mock import patch


class TestSettings:
    """Tests for the Settings class."""

    def _make_settings_class(self):
        """Create a fresh Settings class to avoid import-time side effects."""
        from importlib import import_module, reload
        mod = import_module("config.settings")
        reload(mod)
        return mod.Settings

    def test_default_model_name(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_NAME", None)
            Settings = self._make_settings_class()
            assert Settings.MODEL_NAME == "llama-3.3-70b-versatile"

    def test_default_temperature(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TEMPERATURE", None)
            Settings = self._make_settings_class()
            assert Settings.TEMPERATURE == 0.0

    def test_custom_temperature_from_env(self):
        with patch.dict(os.environ, {"TEMPERATURE": "0.7"}):
            Settings = self._make_settings_class()
            assert Settings.TEMPERATURE == 0.7

    def test_custom_model_name_from_env(self):
        with patch.dict(os.environ, {"MODEL_NAME": "custom-model"}):
            Settings = self._make_settings_class()
            assert Settings.MODEL_NAME == "custom-model"

    def test_groq_api_key_from_env(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}):
            Settings = self._make_settings_class()
            assert Settings.GROQ_API_KEY == "test-key-123"

    def test_validate_raises_when_no_api_key(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}):
            Settings = self._make_settings_class()
            Settings.GROQ_API_KEY = ""
            with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
                Settings.validate()

    def test_validate_returns_true_when_api_key_set(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "valid-key"}):
            Settings = self._make_settings_class()
            Settings.GROQ_API_KEY = "valid-key"
            assert Settings.validate() is True

    def test_categories_list(self):
        Settings = self._make_settings_class()
        assert Settings.CATEGORIES == ["Technical", "Billing", "General"]

    def test_sentiments_list(self):
        Settings = self._make_settings_class()
        assert Settings.SENTIMENTS == ["Positive", "Negative", "Neutral"]
