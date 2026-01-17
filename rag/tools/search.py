from langchain.tools import ToolRuntime, tool
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI

from rag.retrievers.drift import drift_search
from rag.schema.agent import RAGContext
from rag.utils.retrievers import collect_answers

llm: ChatOpenAI = None
vector_store: Neo4jVector = None


@tool(
    name_or_callable="search_knowledge_base",
    description="Retrieves specific facts related to the user's query. Use this tool when the user asks for information, evidence, or details that require external lookups.",
)
def search_knowledge_base(query: str, runtime: ToolRuntime[RAGContext]) -> str:
    drift_config = runtime.context.drift_config
    root = drift_search(query, llm, vector_store, drift_config)
    facts = collect_answers(root)
    # clean up whitespaces and ignore empty string
    formatted_facts = [f"- {fact}" for fact in facts if fact.strip()]
    if formatted_facts:
        facts_str = "### Relevant Facts\n" + "\n".join(formatted_facts)
    else:
        facts_str = "No relevant facts found."
    return facts_str
