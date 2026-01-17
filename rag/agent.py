from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from rag.prompts.agent import SYSTEM_PROMPT
from rag.schema.agent import RAGContext
from rag.tools.search import search_knowledge_base


def create_rag_agent(llm: BaseChatModel):
    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT.invoke({}).to_string(),
        tools=[search_knowledge_base],
        context_schema=RAGContext,
    )
    return agent
