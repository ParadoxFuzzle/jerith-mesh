"""The gateway chat model: ChatOpenAI pinned to the local mesh node.

Everything else stays standard LangChain — the node speaks the
OpenAI-compatible protocol, so no custom transport is needed.
"""

import os

import requests
from langchain_openai import ChatOpenAI

DEFAULT_GATEWAY = "http://127.0.0.1:8900"
_PROBE_TIMEOUT = 5


class JerithChat(ChatOpenAI):
    """ChatOpenAI bound to the local jerith-mesh gateway.

    The gateway needs no API key on loopback; streaming, tool calling
    and the model router all behave like any OpenAI-compatible backend.

        chat = JerithChat()                       # gateway's first chat model
        deep = JerithChat(model="qwen3:8b")       # pin the thinking tier

    With no `model` given, the gateway's /v1/models is consulted once at
    construction (first chat-mode entry wins) — a 'jerith-mesh'
    placeholder is deliberately NOT sent: real runtimes (Ollama,
    LiteLLM) reject it as an invalid model name.
    """

    def __init__(self, *, model: str | None = None, **kwargs) -> None:
        self._base_url = os.environ.get("JERITH_GATEWAY", DEFAULT_GATEWAY).rstrip("/")
        kwargs.setdefault("base_url", f"{self._base_url}/v1")
        # Loopback endpoint: the key is a placeholder, never a secret.
        kwargs.setdefault("api_key", "jerith-local")
        kwargs.setdefault(
            "model",
            model or os.environ.get("JERITH_MODEL") or self._default_model(),
        )
        super().__init__(**kwargs)

    def _default_model(self) -> str:
        """First chat-mode model the gateway reports, or a loud error."""
        try:
            resp = requests.get(
                f"{self._base_url}/v1/models", timeout=_PROBE_TIMEOUT
            )
            resp.raise_for_status()
            entries = resp.json().get("data") or []
            chats = [e["id"] for e in entries if e.get("mode", "chat") == "chat"]
            if chats:
                return chats[0]
        except Exception as exc:  # noqa: BLE001 — replaced by the raise below
            raise RuntimeError(
                f"cannot reach the jerith-mesh gateway at {self._base_url} "
                f"to pick a model ({exc}); construct JerithChat(model='…') explicitly"
            ) from exc
        raise RuntimeError(
            f"the gateway at {self._base_url} lists no chat models; "
            "construct JerithChat(model='…') explicitly"
        )
