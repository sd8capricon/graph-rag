from typing import Annotated

from fastapi import APIRouter, Depends
from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from api.dependencies.agent import get_rag_agent
from api.dependencies.llm import get_llm
from api.dependencies.vector_store import provide_vector_store
from api.enums.vector_store import VectorStoreName
from api.schema.chat import ChatRequest
from rag.schema.agent import RAGContext
from rag.types.agent import RAGAgent

router = APIRouter()


@router.post("")
async def chat(
    payload: ChatRequest,
    agent: Annotated[RAGAgent, Depends(get_rag_agent)],
    llm: Annotated[BaseChatModel, Depends(get_llm)],
    vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.community))
    ],
):
    res = agent.invoke(
        {"messages": [HumanMessage(payload.query)]},
        context=RAGContext(
            drift_config={"top_k": 5, "max_depth": 2, "max_follow_ups": 3},
            llm=llm,
            vector_store=vector_store,
        ),
    )
    return {"message": res["messages"]}
