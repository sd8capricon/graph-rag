from langchain.tools import ToolRuntime, tool

from rag.retrievers.drift import adrift_search, drift_search
from rag.retrievers.similarity import asimilarity_search
from rag.schema.agent import RAGContext
from rag.utils.retrievers import collect_answers


def _format_facts(facts: list[str]) -> str:
    formatted = [f"- {fact}" for fact in facts if fact.strip()]
    if formatted:
        return "### Relevant Facts\n" + "\n".join(formatted)
    return "No relevant facts found."


def _get_drift_config(runtime: ToolRuntime[RAGContext]):
    ctx = runtime.context
    return ctx.drift_config, ctx.llm, ctx.commuunity_vector_store


@tool(
    name_or_callable="detail_search_knowledge_base",
    description="Retrieves specific facts related to the user's query. Use this tool when the user asks for information, evidence, or details that require external lookups.",
)
def search_knowledge_base(query: str, runtime: ToolRuntime[RAGContext]) -> str:
    drift_config, llm, vector_store = _get_drift_config(runtime)
    root = drift_search(query, llm, vector_store, drift_config)
    facts = collect_answers(root)
    return _format_facts(facts)


@tool(
    name_or_callable="detail_search_knowledge_base",
    description="Retrieves specific facts related to the user's query. Use this tool when the user asks for detailed information, evidence, or details that require external lookups.",
)
async def asearch_knowledge_base(query: str, runtime: ToolRuntime[RAGContext]) -> str:
    drift_config, llm, vector_store = _get_drift_config(runtime)
    root = await adrift_search(query, llm, vector_store, drift_config)
    facts = collect_answers(root)
    return _format_facts(facts)


@tool(
    name_or_callable="similarity_search",
    description="Retrieves facts related to the user's query. Use this to give an initial answer to the user's query",
)
async def asimilarity_search(query: str, runtime: ToolRuntime[RAGContext]):
    vector_store = runtime.context.lexical_vector_store
    return asimilarity_search(query, vector_store)
