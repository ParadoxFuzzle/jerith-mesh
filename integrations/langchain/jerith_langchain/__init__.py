"""jerith-langchain — use the local jerith-mesh node from LangChain.

The node exposes an OpenAI-compatible gateway on localhost (default
http://127.0.0.1:8900), so `ChatOpenAI` works against it with no key.
This package is a thin convenience layer: the base URL default, plus
@tool wrappers around the node's strictly-local skill endpoints
(memory, lore, web search) so agents can use them directly.

Install (with the mesh's managed tools venv):

    jerith stack install langchain   # langchain + langchain-openai
    jerith stack install langgraph   # for the agent example
    jerith stack install langsmith   # optional tracing

Or any Python 3.10+ environment:

    pip install -e integrations/langchain

Env vars:
    JERITH_GATEWAY   base URL of the node's gateway (default http://127.0.0.1:8900)
    JERITH_MODEL     model the node should resolve (default: node default)
"""

from .chat import JerithChat
from .skills import memory_search, memory_store, lore_search, web_search

__all__ = [
    "JerithChat",
    "memory_search",
    "memory_store",
    "lore_search",
    "web_search",
]
__version__ = "0.1.0"
