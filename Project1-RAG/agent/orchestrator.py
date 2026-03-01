"""
LangGraph ReAct agent wired to Claude via LangChain's ChatAnthropic.

The agent follows the Thought → Action → Observation loop:
  1. Claude decides which tool to call (if any).
  2. LangGraph executes the tool node.
  3. The observation is fed back to Claude.
  4. Loop repeats until Claude issues a final answer.

Conversation history is persisted in the checkpointer (MemorySaver by
default), keyed by `thread_id` in the run config.
"""
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent

from config import settings

SYSTEM_PROMPT = """You are a helpful RAG (Retrieval-Augmented Generation) assistant.

You have access to the following tools:
- retrieve_docs  : search the private knowledge base
- web_search     : search the web for recent / real-time information
- summarize_text : condense a long passage to its key points

Guidelines:
1. Always try `retrieve_docs` first for any domain-specific question.
2. If the knowledge base is insufficient, fall back to `web_search`.
3. Cite your sources — mention document names, page numbers, or URLs.
4. If you are uncertain, say so. Never fabricate facts.
5. Keep answers clear, well-structured, and concise.
"""


def create_rag_agent(tools: list, memory):
    """
    Build and return a compiled LangGraph ReAct agent.

    Args:
        tools:  list of LangChain tool callables
        memory: a LangGraph checkpointer (MemorySaver, PostgresSaver, …)

    Returns:
        A compiled StateGraph ready to call with `.invoke()` or `.stream()`.
    """
    model = ChatAnthropic(
        model=settings.claude_model,
        anthropic_api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
    )
    return create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory,
        state_modifier=SYSTEM_PROMPT,
    )
