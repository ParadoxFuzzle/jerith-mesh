"""A tool-using LangGraph agent backed by the local jerith-mesh node.

Setup:

    jerith stack install langchain   # langchain + langchain-openai
    jerith stack install langgraph   # this example's runtime

Optional LangSmith observability (no code changes — the standard env
vars work because the run is an ordinary LangChain run):

    export LANGSMITH_TRACING=true
    export LANGSMITH_API_KEY=lsv2_...      # your key; never commit it

Run:

    python integrations/langchain/langgraph_agent.py "what's on the mesh right now?"

Env:
    JERITH_GATEWAY   gateway base URL (default http://127.0.0.1:8900)
    JERITH_MODEL     model id (default: the node's routing default)
"""

import sys

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from jerith_langchain.chat import JerithChat
from jerith_langchain.skills import lore_search, memory_search, memory_store, web_search

TOOLBOX = [memory_store, memory_search, lore_search, web_search]

# With the node's tool tier pinned (`jerith config set
# runtime_tool_model=nemotron-3-nano:4b`), the gateway reroutes every
# tool-bearing request (like these) to the reliable tool caller
# automatically — no JERITH_MODEL pin needed. Without the pin, note the
# measured live-runtime behavior (2026-10-16): qwen3:8b thinking chokes
# tool-call parsing ({}`) — pin a reliable tool-calling model instead:
#     JERITH_MODEL=nemotron-3-nano:4b python3 langgraph_agent.py …
llm = JerithChat(temperature=0.1).bind_tools(TOOLBOX)


SYSTEM = (
    "You are the jerith-mesh node's agent. Use the tools to gather facts, "
    "then give a short final answer in plain text. Call a tool at most "
    "once per question — when the tool result is in, answer; never repeat "
    "a tool call you already made."
)


def chatbot(state: MessagesState):
    """One model turn — the node's model-router tiers apply per request."""
    messages = [("system", SYSTEM), *state["messages"]]
    return {"messages": [llm.invoke(messages)]}


def build():
    """The canonical LangGraph tool loop: start → chatbot → (tools →)* END."""
    graph = StateGraph(MessagesState)
    graph.add_node("chatbot", chatbot)
    graph.add_node("tools", ToolNode(TOOLBOX))
    graph.add_edge("__start__", "chatbot")
    # tools_condition: route to "tools" when the model asked for one,
    # END when it produced a final answer.
    graph.add_conditional_edges("chatbot", tools_condition)
    graph.add_edge("tools", "chatbot")
    return graph.compile()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "search my memory for anything about the mesh"
    result = build().invoke({"messages": [("user", question)]})
    print(result["messages"][-1].content)
