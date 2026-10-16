# jerith-mesh × LangChain

The mesh node ships an **OpenAI-compatible gateway** on
`http://127.0.0.1:8900/v1` (`JERITH_GATEWAY` to override), so LangChain
talks to it with the stock `ChatOpenAI` — no key, no custom transport.
This package adds two small things: a pinned `JerithChat` and `@tool`
wrappers around the node's **strictly-local skill endpoints** (memory,
lore, web search) so agents can use them directly.

## Quickstart

```bash
# inside the repo you are building agents in:
pip install -e /path/to/jerith-mesh/integrations/langchain

# or via the mesh's managed tools venv:
jerith stack install langchain && jerith stack install langgraph
```

```python
from jerith_langchain import JerithChat, memory_search, web_search

chat = JerithChat()                     # node default model
deep = JerithChat(model="qwen3:8b")     # pin the thinking tier

print(chat.invoke("hello").content)      # ordinary LangChain Runnable
```

## The agent example

`langgraph_agent.py` wires a LangGraph tool loop over the skill tools
and runs it against the node's current routing default (or `JERITH_MODEL`):

```bash
python integrations/langchain/langgraph_agent.py "what's on the mesh right now?"
```

## Model tiers (how the node picks models)

The gateway routes each request: image content → the **vision** tier
(default `minicpm-v4.6:1b`), complex/long prompts and explicit `[deep]`
tags → the **thinking** tier (default `qwen3:8b`), everything else keeps
the caller's model. Pin a model per `JerithChat(...)` call to opt out of
rerouting entirely.

Tier picks are installed/pulled with:

```bash
jerith stack list --models   # the curated small-model catalog
jerith stack pull qwen3:8b   # or run `jerith setup` and pick there
```

## LangSmith (optional observability)

Standard env vars, nothing in code:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...   # your key — never commit it
```

With no key set, nothing leaves the machine except requests to the
loopback gateway.

## Postgres + pgvector memory backend

The node's memory is strictly local. To move it from SQLite to Postgres:

```bash
jerith stack install postgres   # server + pgvector extension
jerith config set memory_backend=postgres \
    memory_dsn=postgres://postgres@127.0.0.1:5432/jerith?sslmode=disable
jerith memory migrate            # copy the existing SQLite rows over
```

Agent-side nothing changes: the `memory_*` tools hit the node, the node
hits its own store.
