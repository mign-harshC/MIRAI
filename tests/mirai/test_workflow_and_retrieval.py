"""Tests for MIRAI workflow validation and retrieval response handling.

Modified for the MIRAI project, 2026.
"""

from qa_manager.base_agent_tag import RetrievalTool, is_workflow_valid


def test_workflow_validation() -> None:
    assert is_workflow_valid("QDS")
    assert is_workflow_valid("QDP")
    assert is_workflow_valid("QR,R,DS,AG")
    assert is_workflow_valid("R,AG")
    assert not is_workflow_valid("QDS,AG")
    assert not is_workflow_valid("DS,R,AG")
    assert not is_workflow_valid("QR,R,R,AG")


def test_retrieval_tool_uses_configured_endpoint(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return [{"top_k_docs": [{"id": "doc-1", "text": "evidence"}]}]

    called = {}

    def fake_post(url, headers, data, timeout):
        called.update(url=url, headers=headers, data=data, timeout=timeout)
        return Response()

    monkeypatch.setattr("qa_manager.base_agent_tag.requests.post", fake_post)
    tool = RetrievalTool(["http://127.0.0.1:8000/search"])
    docs = tool.query("question", 1)

    assert docs == [{"id": "doc-1", "text": "evidence"}]
    assert called["url"] == "http://127.0.0.1:8000/search"
    assert called["timeout"] == 5
