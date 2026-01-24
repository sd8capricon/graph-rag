from typing import Annotated, Callable

from fastapi import APIRouter, Depends
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI

from api.dependencies.llm import get_llm
from api.dependencies.vector_store import provide_vector_store
from api.enums.vector_store import VectorStoreName
from rag.agent import create_rag_agent

router = APIRouter()


@router.post("")
async def chat(
    llm: Annotated[ChatOpenAI, Depends(get_llm)],
    vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.community))
    ],
):
    print(type(vector_store))
    agent = create_rag_agent(llm)
    return {"message": "chat"}
