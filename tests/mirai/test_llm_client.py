"""Unit tests for the MIRAI OpenAI-compatible client.

Modified for the MIRAI project, 2026.
"""

from types import SimpleNamespace

import pytest

from qa_manager.llm_api import LLM_Client


class FakeCompletions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(role="assistant", content="MIRAI test response")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_query_model_normalizes_response() -> None:
    fake = FakeClient()
    client = LLM_Client(model_name="test-model", base_url="http://localhost:8080/v1", client=fake)
    choices, elapsed = client.query_model(
        [{"role": "user", "content": "hello"}], temperature=0.2, max_tokens=16
    )

    assert choices == [{"message": {"role": "assistant", "content": "MIRAI test response"}}]
    assert elapsed >= 0
    assert fake.chat.completions.request["model"] == "test-model"
    assert fake.chat.completions.request["max_tokens"] == 16


def test_query_requires_api_or_local_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    client = LLM_Client(model_name="test-model", client=FakeClient())
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        client.query_model([{"role": "user", "content": "hello"}])


def test_empty_messages_are_rejected() -> None:
    client = LLM_Client(model_name="test-model", base_url="http://localhost:8080/v1", client=FakeClient())
    with pytest.raises(ValueError, match="messages"):
        client.query_model([])

