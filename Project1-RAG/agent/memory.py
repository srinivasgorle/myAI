"""
Conversation memory for the RAG agent.

Uses LangGraph's MemorySaver as an in-process checkpointer so the agent
can maintain multi-turn conversation state per session (thread_id).

For production, swap MemorySaver for a PostgresSaver or RedisSaver.
"""
from langgraph.checkpoint.memory import MemorySaver


def create_memory() -> MemorySaver:
    """Return a new in-memory checkpointer instance."""
    return MemorySaver()
