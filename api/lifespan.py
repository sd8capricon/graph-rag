from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.dependencies.agent import set_agent
from api.dependencies.database import close_neo4j_connection, set_neo4j_connection
from api.dependencies.knowledge_base import set_kb_service
from api.dependencies.llm import get_llm
from api.dependencies.vector_store import set_vector_store_service
from api.enums.vector_store import VectorStoreName
from api.schema.vector_store import VectorStoreConfig
from api.services.vector_store import VectorStoreService
from common.graph.client import GraphClient
from common.graph.config import client_config
from common.services.knowledge_base import KnowledgeBaseService
from rag.agent import create_rag_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize neo4j connection
    neo4j_connection = GraphClient(**client_config())
    set_neo4j_connection(neo4j_connection)
    # Initialize knowledge base service
    kb_service = KnowledgeBaseService(neo4j_connection)
    set_kb_service(kb_service)
    # Initialize vector stores on startup
    vector_store_service = VectorStoreService()
    configs = [
        VectorStoreConfig(
            name=VectorStoreName.community,
            node_label="Community",
            text_property="summary",
            index_name="community_vector_index",
        ),
        VectorStoreConfig(
            name=VectorStoreName.lexical,
            text_property="text",
            index_name="vector_index",
            retrieval_query="RETURN node.text AS text, score, node {.*, text: Null, embedding:Null} as metadata",
        ),
    ]
    await vector_store_service.initialize(configs)
    set_vector_store_service(vector_store_service)
    # Initialize agent
    llm = get_llm()
    agent = create_rag_agent(llm)
    set_agent(agent)
    yield
    close_neo4j_connection()
