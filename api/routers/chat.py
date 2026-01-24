from typing import Annotated

from fastapi import APIRouter, Depends
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from api.dependencies.agent import get_rag_agent
from api.dependencies.vector_store import provide_vector_store
from api.enums.vector_store import VectorStoreName
from rag.types.agent import RAGAgent

router = APIRouter()


@router.post("")
async def chat(
    agent: Annotated[RAGAgent, Depends(get_rag_agent)],
    vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.community))
    ],
):
    print(type(vector_store))
    return {"message": "chat"}
