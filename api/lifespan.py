import os
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

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

SQL_LITE_DB = os.getenv("SQL_LITE_DB", "database.db")


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


async def init_agent(checkpointer: BaseCheckpointSaver | None = None) -> RAGAgent:
    """
    Initialize and register the RAG agent.

    Creates a retrieval-augmented generation (RAG) agent using the
    configured LLM and registers it for dependency injection.

    Returns:
        RAGAgent: The initialized RAG agent instance.
    """
    llm = get_llm()
    agent = create_rag_agent(llm, checkpointer)
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
    Manage the FastAPI application startup and shutdown lifecycle.

    Startup sequence:
        - Configure application logging
        - Initialize the Neo4j database connection
        - Initialize the knowledge base service
        - Initialize vector stores
        - Open an asynchronous SQLite connection
        - Create the AsyncSqliteSaver checkpointer
        - Initialize and register the RAG agent

    Shutdown sequence:
        - Close the asynchronous SQLite connection
        - Perform application shutdown and resource cleanup

    Notes:
        - The SQLite connection is kept open for the lifetime of the application
          and is explicitly closed during shutdown to prevent event loop hangs.
        - All initialization steps must complete successfully before the
          application starts accepting requests.

    Args:
        app (FastAPI): The FastAPI application instance.

    Yields:
        None: Control is yielded to FastAPI for the duration of the application runtime.
    """
    setup_logger()
    neo4j_connection = init_neo4j()
    init_kb_service(neo4j_connection)
    await init_vector_store()
    sqlite_conn = await aiosqlite.connect(SQL_LITE_DB)
    checkpointer = AsyncSqliteSaver(sqlite_conn)
    await init_agent(checkpointer)
    yield
    sqlite_conn.close()
    shutdown()
