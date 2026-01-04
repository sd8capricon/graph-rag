from typing import TypedDict

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.vectorstores import VectorStore

from rag.prompts.retrievers import HYDE_SYSTEM_PROMPT, PRIMER_SEARCH_PROMPT
from rag.retrievers.vector import vector_search
from rag.schema.retrievers import Answer, Node


class DriftConfig(TypedDict):
    top_k: int = 5
    max_depth: int = 2
    max_follow_ups: int = 3


__all__ = ["drift_search"]

answer_parser = PydanticOutputParser(pydantic_object=Answer)


def expand_query(query: str, llm: BaseChatModel) -> str:
    system_prompt = HYDE_SYSTEM_PROMPT.invoke({}).to_string()
    res = llm.invoke([SystemMessage(system_prompt), HumanMessage(f"Question {query}")])
    return query + res.content


def primer_search(
    query: str,
    top_communities: list[Document],
    llm: BaseChatModel,
    max_follow_ups: int = 3,
) -> tuple[str, list[str]]:
    context = "Context:\n\n" + "\n\n---\n\n".join(
        r.page_content for r in top_communities
    )
    system_prompt = PRIMER_SEARCH_PROMPT.invoke(
        {
            "follow_up_count": max_follow_ups,
            "output_instructions": answer_parser.get_format_instructions(),
        }
    ).to_string()

    res = llm.invoke(
        [SystemMessage(system_prompt), SystemMessage(context), HumanMessage(query)]
    )
    answer: Answer = answer_parser.invoke(res)

    return answer.body, answer.follow_up_questions


def drift_search(
    query: str,
    llm: BaseChatModel,
    vector_store: VectorStore,
    config: DriftConfig,
    depth: int = 1,
):
    # Step 1: Prepare initial query representation
    expanded_query = expand_query(query, llm)

    # Step 2: Primer — global search for high-level context
    top_communities = vector_search(expanded_query, vector_store, config["top_k"])
    initial_answer, follow_up_questions = primer_search(
        query, top_communities, llm, config["max_follow_ups"]
    )

    node = Node(query=query, answer=initial_answer)

    if depth >= config.get("max_depth", 3):
        return node

    for follow_up in follow_up_questions:
        child = drift_search(follow_up, llm, vector_store, config, depth + 1)
        node.add_child(child)

    return node
