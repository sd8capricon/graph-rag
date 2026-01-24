from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from rag.prompts.agent import SYSTEM_PROMPT
from rag.schema.agent import RAGContext
from rag.tools.search import asearch_knowledge_base
from rag.types.agent import RAGAgent


def create_rag_agent(llm: BaseChatModel) -> RAGAgent:
    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT.invoke({}).to_string(),
        tools=[asearch_knowledge_base],
        context_schema=RAGContext,
    )
    return agent
