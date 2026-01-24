from typing import TypeAlias

from langchain.agents import AgentState
from langchain.agents.middleware.types import _InputAgentState, _OutputAgentState
from langgraph.graph.state import CompiledStateGraph

from rag.schema.agent import RAGContext

RAGAgent: TypeAlias = CompiledStateGraph[
    AgentState, RAGContext, _InputAgentState, _OutputAgentState
]
