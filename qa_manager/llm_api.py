"""Environment-configured OpenAI-compatible client used by MIRAI roles.

Modified for the MIRAI project, 2026.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
from openai import OpenAI


MODEL_ALIASES = {
    "gpt4o": "gpt-4o",
    "gpt4o-mini": "gpt-4o-mini",
}


class LLM_Client:
    """Small compatibility wrapper around an OpenAI-compatible chat API.

    Credentials and endpoints are read from ``OPENAI_API_KEY`` and
    ``OPENAI_BASE_URL``. No secret is stored in source code. Template paths are
    optional and retained for compatibility with existing role constructors.
    """

    def __init__(
        self,
        model_name: str | None = None,
        system_jinja2_path: str | None = None,
        user_jinja2_path: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        client: Any | None = None,
    ) -> None:
        requested_model = model_name or os.getenv("MIRAI_LLM_MODEL", "gpt-4o-mini")
        self.model_name = MODEL_ALIASES.get(requested_model, requested_model)
        self.system_jinja2_path = system_jinja2_path
        self.user_jinja2_path = user_jinja2_path
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.timeout = timeout or float(os.getenv("MIRAI_LLM_TIMEOUT", "60"))
        self._client = client or OpenAI(
            api_key=self.api_key or "not-required",
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @staticmethod
    def _render_template(path: str | None, **context: Any) -> str | None:
        if not path:
            return None
        template_path = Path(path).expanduser()
        if not template_path.is_file():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")
        environment = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=False)
        try:
            template = environment.get_template(template_path.name)
        except TemplateNotFound as exc:
            raise FileNotFoundError(f"Prompt template not found: {template_path}") from exc
        return template.render(**context)

    def query_model(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> tuple[list[dict[str, dict[str, str]]], float]:
        """Return normalized choices and request latency.

        The method intentionally accepts extra keyword arguments for
        compatibility with role code and alternate OpenAI-compatible servers.
        """

        if not messages:
            raise ValueError("messages must not be empty")
        if not self.api_key and not self.base_url:
            raise RuntimeError(
                "Set OPENAI_API_KEY for the public API or OPENAI_BASE_URL for a local compatible endpoint."
            )

        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "n": n,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        request.update(kwargs)

        started = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started

        choices: list[dict[str, dict[str, str]]] = []
        for choice in response.choices:
            content = choice.message.content or ""
            choices.append({"message": {"role": choice.message.role or "assistant", "content": content}})
        return choices, elapsed
