"""Skill tools: the node's strictly-local endpoints as LangChain tools.

Each tool POSTs to the local gateway's /v1/skills/* endpoints — memory
and lore never leave the node; web search is the node's own (Brave-backed,
key-holding) service, so agent code never sees any API keys.
"""

import json
import os
import urllib.request

from langchain_core.tools import tool

DEFAULT_GATEWAY = "http://127.0.0.1:8900"


def _gateway() -> str:
    return os.environ.get("JERITH_GATEWAY", DEFAULT_GATEWAY).rstrip("/")


def _post(path: str, payload: dict) -> str:
    """POST JSON to a gateway skill endpoint and return the raw body.

    Any failure returns a short error string — a tool error must pierce
    the agent loop as text, not crash the run.
    """
    req = urllib.request.Request(
        f"{_gateway()}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode()
    except Exception as exc:  # noqa: BLE001 — surfaced to the agent as text
        return f"error: {exc}"


@tool
def memory_store(text: str) -> str:
    """Remember a piece of information in the node's local memory.

    Use when the user asks you to save or remember something. Entries
    stay on this machine; nothing is uploaded anywhere.
    """
    return _post("/v1/skills/memory/embed", {"text": text})


@tool
def memory_search(query: str, top_k: int = 5) -> str:
    """Search the node's local memory for past entries relevant to `query`.

    Use before answering questions the user may have told you about
    before (preferences, facts, past tasks).
    """
    return _post("/v1/skills/memory/search", {"query": query, "top_k": top_k})


@tool
def lore_search(query: str, top_k: int = 4) -> str:
    """Search the jerith lore vault (project canon) for `query`.

    Use for questions about the mesh itself, its world or its rules.
    """
    return _post("/v1/skills/lore/search", {"query": query, "top_k": top_k})


@tool
def web_search(query: str, count: int = 5) -> str:
    """Search the web via the node's Brave-backed search skill.

    Use for anything requiring fresh information. The node holds the
    search API key; callers never do.
    """
    return _post("/v1/skills/web/search", {"query": query, "count": count})
