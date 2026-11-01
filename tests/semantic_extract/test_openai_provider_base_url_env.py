"""``OpenAIProvider`` must treat ``OPENAI_BASE_URL`` as a custom endpoint.

The openai SDK reads ``OPENAI_BASE_URL`` on its own, so requests already went
to the configured server, but the provider's ``base_url`` attribute stayed
``None``. Every custom-endpoint branch keys off that attribute: with it unset
the provider sent ``response_format={"type": "json_object"}`` (LM Studio
answers ``400 'response_format.type' must be 'json_schema' or 'text'``) and
``instructor`` ran in ``Mode.TOOLS`` instead of ``Mode.JSON``.
"""

from __future__ import annotations

import pytest

from semantica.semantic_extract.providers import OpenAIProvider


@pytest.fixture
def _no_openai_client(monkeypatch):
    # Keep the test offline: the client is never needed to inspect base_url.
    monkeypatch.setattr(OpenAIProvider, "_init_client", lambda self: None)


def test_env_base_url_becomes_custom_endpoint(monkeypatch, _no_openai_client):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "lm-studio")

    provider = OpenAIProvider(model="local-model")

    assert provider.base_url == "http://localhost:1234/v1"


def test_explicit_base_url_wins_over_env(monkeypatch, _no_openai_client):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    provider = OpenAIProvider(model="m", base_url="https://gateway.example/v1")

    assert provider.base_url == "https://gateway.example/v1"


def test_no_env_keeps_standard_openai(monkeypatch, _no_openai_client):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    provider = OpenAIProvider(model="gpt-4o-mini")

    assert provider.base_url is None


def test_empty_env_is_ignored(monkeypatch, _no_openai_client):
    monkeypatch.setenv("OPENAI_BASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "key")

    provider = OpenAIProvider(model="gpt-4o-mini")

    assert provider.base_url is None
