from fastapi.exceptions import HTTPException
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from api.schema.vector_store import VectorStoreConfig
from common.embedding import get_embedding
from common.graph.config import neo4j_config


class VectorStoreService:
    """Service for managing vector store initialization and access."""

    def __init__(self):
        self._stores: dict[str, Neo4jVector] = {}

    async def initialize(self, configs: list[VectorStoreConfig]) -> None:
        """Initialize vector stores from configurations."""
        for config in configs:
            self._stores[config.name] = self._build_store(config)

    def _build_store(config: VectorStoreConfig) -> Neo4jVector:
        kwargs = {
            "embedding": get_embedding(),
            "text_node_property": config.text_property,
            "embedding_node_property": "embedding",
            "index_name": config.index_name,
            **neo4j_config(),
        }
        if config.node_label:
            kwargs["node_label"] = config.node_label
        if config.retrieval_query:
            kwargs["retrieval_query"] = config.retrieval_query
        return Neo4jVector(**kwargs)

    def get_store(self, name: str) -> Neo4jVector:
        if name not in self._stores:
            raise HTTPException(500, f"Vector store '{name}' not initialized")
        return self._stores[name]
