from rag.types.agent import RAGAgent

_agent: RAGAgent | None = None


def set_agent(agent: RAGAgent):
    global _agent
    _agent = agent


def get_rag_agent():
    if not _agent:
        raise ValueError("RAG Agent not initialized")
    return _agent
