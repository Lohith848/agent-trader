"""Tests for the canonical provider->env-var mapping and the CLI key-prompt helper."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from tradingagents.llm_clients.api_key_env import PROVIDER_API_KEY_ENV, get_api_key_env

# ---- Mapping coverage -----------------------------------------------------


def test_every_select_llm_provider_choice_has_an_entry():
    """select_llm_provider() must not present a provider the mapping doesn't know about."""
    expected = {
        "google", "groq", "openrouter", "nvidia", "ollama",
    }
    assert expected.issubset(PROVIDER_API_KEY_ENV.keys())


@pytest.mark.parametrize(
    "provider,env_var",
    [
        ("google",     "GOOGLE_API_KEY"),
        ("groq",       "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
        ("nvidia",     "NVIDIA_API_KEY"),
    ],
)
def test_known_providers_resolve(provider, env_var):
    assert get_api_key_env(provider) == env_var


def test_ollama_has_no_key():
    assert get_api_key_env("ollama") is None


def test_unknown_provider_returns_none():
    assert get_api_key_env("not-a-real-provider") is None


def test_case_insensitive_lookup():
    assert get_api_key_env("Google") == "GOOGLE_API_KEY"
    assert get_api_key_env("GROQ") == "GROQ_API_KEY"
    assert get_api_key_env("OpenRouter") == "OPENROUTER_API_KEY"
    assert get_api_key_env("Nvidia") == "NVIDIA_API_KEY"


# ---- ensure_api_key behavior ---------------------------------------------


@pytest.fixture
def cli_utils(monkeypatch):
    """Import cli.utils with a fresh environment so module-level state is consistent."""
    import importlib

    import cli.utils as cli_utils_module
    return importlib.reload(cli_utils_module)


def test_ensure_api_key_returns_existing(monkeypatch, cli_utils):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-already-set")
    result = cli_utils.ensure_api_key("groq")
    assert result == "gsk-already-set"


def test_ensure_api_key_no_op_for_ollama(monkeypatch, cli_utils):
    # Even with no env var set, ollama should not prompt and should return None.
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with patch.object(cli_utils, "questionary") as mock_q:
        result = cli_utils.ensure_api_key("ollama")
    assert result is None
    mock_q.password.assert_not_called()


def test_ensure_api_key_unknown_provider_no_prompt(monkeypatch, cli_utils):
    with patch.object(cli_utils, "questionary") as mock_q:
        result = cli_utils.ensure_api_key("totally-fake-provider")
    assert result is None
    mock_q.password.assert_not_called()


def test_ensure_api_key_prompts_and_writes_to_env(monkeypatch, tmp_path, cli_utils):
    """When key is missing, user-pasted value must be written to .env AND os.environ."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    fake_prompt = type("P", (), {"ask": staticmethod(lambda: "nvapi-test")})()
    with patch.object(cli_utils.questionary, "password", return_value=fake_prompt):
        result = cli_utils.ensure_api_key("nvidia")

    assert result == "nvapi-test"
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-test"
    env_file = tmp_path / ".env"
    assert env_file.exists()
    assert "NVIDIA_API_KEY" in env_file.read_text()
    assert "nvapi-test" in env_file.read_text()


def test_ensure_api_key_user_cancels_returns_none(monkeypatch, tmp_path, cli_utils):
    """Empty prompt response (user cancelled) must not write to .env."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    fake_prompt = type("P", (), {"ask": staticmethod(lambda: None)})()
    with patch.object(cli_utils.questionary, "password", return_value=fake_prompt):
        result = cli_utils.ensure_api_key("groq")

    assert result is None
    assert "GROQ_API_KEY" not in os.environ
    env_file = tmp_path / ".env"
    if env_file.exists():
        assert "GROQ_API_KEY" not in env_file.read_text()


def test_ensure_api_key_updates_existing_env_file(monkeypatch, tmp_path, cli_utils):
    """An existing .env with other keys must be preserved on writeback."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("GOOGLE_API_KEY=key-existing\nOTHER=value\n")

    fake_prompt = type("P", (), {"ask": staticmethod(lambda: "sk-openrouter-new")})()
    with patch.object(cli_utils.questionary, "password", return_value=fake_prompt):
        cli_utils.ensure_api_key("openrouter")

    content = env_file.read_text()
    assert "GOOGLE_API_KEY" in content and "key-existing" in content
    assert "OTHER=value" in content
    assert "OPENROUTER_API_KEY" in content and "sk-openrouter-new" in content
