from langchain.tools import ToolRuntime, tool

from rag.retrievers.drift import adrift_search, drift_search
from rag.schema.agent import RAGContext
from rag.utils.retrievers import collect_answers


def _format_facts(facts: list[str]) -> str:
    formatted = [f"- {fact}" for fact in facts if fact.strip()]
    if formatted:
        return "### Relevant Facts\n" + "\n".join(formatted)
    return "No relevant facts found."


def _get_runtime_components(runtime: ToolRuntime[RAGContext]):
    ctx = runtime.context
    return ctx.drift_config, ctx.llm, ctx.vector_store


@tool(
    description="Retrieves specific facts related to the user's query. Use this tool when the user asks for information, evidence, or details that require external lookups.",
)
def search_knowledge_base(query: str, runtime: ToolRuntime[RAGContext]) -> str:
    drift_config, llm, vector_store = _get_runtime_components(runtime)
    root = drift_search(query, llm, vector_store, drift_config)
    facts = collect_answers(root)
    return _format_facts(facts)


@tool(
    name_or_callable="search_knowledge_base",
    description="Retrieves specific facts related to the user's query. Use this tool when the user asks for information, evidence, or details that require external lookups.",
)
async def asearch_knowledge_base(query: str, runtime: ToolRuntime[RAGContext]) -> str:
    drift_config, llm, vector_store = _get_runtime_components(runtime)
    root = await adrift_search(query, llm, vector_store, drift_config)
    facts = collect_answers(root)
    return _format_facts(facts)
