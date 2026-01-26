import logging
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
from common.utils.logger import setup_logger
from rag.agent import create_rag_agent
from rag.types.agent import RAGAgent


def init_neo4j() -> GraphClient:
    """
    Initialize and register the Neo4j graph client.

    Creates a GraphClient using application configuration, registers it
    in the dependency container, and returns the initialized connection.

    Returns:
        GraphClient: The initialized Neo4j graph client.
    """
    neo4j_connection = GraphClient(**client_config())
    set_neo4j_connection(neo4j_connection)
    return neo4j_connection


def init_kb_service(connection: GraphClient) -> KnowledgeBaseService:
    """
    Initialize and register the Knowledge Base service.

    Wraps the provided Neo4j connection in a KnowledgeBaseService and
    registers it for dependency injection.

    Args:
        connection (GraphClient): Active Neo4j graph connection.

    Returns:
        KnowledgeBaseService: The initialized knowledge base service.
    """
    kb_service = KnowledgeBaseService(connection)
    set_kb_service(kb_service)
    return kb_service


async def init_vector_store() -> VectorStoreService:
    """
    Initialize and register the Vector Store service.

    Configures multiple vector stores with their respective schemas and
    initializes them asynchronously, then registers the service for use
    across the application.

    Returns:
        VectorStoreService: The initialized vector store service.
    """
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
            retrieval_query=(
                "RETURN node.text AS text, score, "
                "node {.*, text: Null, embedding:Null} as metadata"
            ),
        ),
    ]
    await vector_store_service.initialize(configs)
    set_vector_store_service(vector_store_service)
    return vector_store_service


def init_agent() -> RAGAgent:
    """
    Initialize and register the RAG agent.

    Creates a retrieval-augmented generation (RAG) agent using the
    configured LLM and registers it for dependency injection.

    Returns:
        RAGAgent: The initialized RAG agent instance.
    """
    llm = get_llm()
    agent = create_rag_agent(llm)
    set_agent(agent)
    return agent


def shutdown():
    """
    Cleanly shut down application resources.

    Closes the Neo4j connection and releases any related resources.
    """
    close_neo4j_connection()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    On startup:
        - Initializes Neo4j connection
        - Sets up the knowledge base service
        - Initializes vector stores
        - Creates and registers the RAG agent

    On shutdown:
        - Closes the Neo4j connection

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded back to FastAPI during app runtime.
    """
    setup_logger()
    neo4j_connection = init_neo4j()
    init_kb_service(neo4j_connection)
    await init_vector_store()
    init_agent()
    yield
    shutdown()
