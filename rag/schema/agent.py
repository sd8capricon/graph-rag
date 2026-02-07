from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from rag.schema.retrievers import DriftConfig


@dataclass
class RAGContext:
    drift_config: DriftConfig
    llm: BaseChatModel
    lexical_vector_store: Neo4jVector | None = None
    commuunity_vector_store: Neo4jVector | None = None
