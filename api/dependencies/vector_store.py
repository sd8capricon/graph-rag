from fastapi.exceptions import HTTPException
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from common.embedding import get_embedding
from common.graph.config import neo4j_config


class VectorStoreService:
    """Service for managing vector store initialization and access."""

    _community_store: Neo4jVector | None = None
    _lexical_store: Neo4jVector | None = None

    async def initialize(self) -> None:
        """Initialize both vector stores."""
        self._community_store = self._build_community_store()
        self._lexical_store = self._build_lexical_store()

    @staticmethod
    def _build_store(
        node_label: str | None = None,
        text_property: str = "text",
        index_name: str = "vector_index",
        retrieval_query: str | None = None,
    ) -> Neo4jVector:
        kwargs = {
            "embedding": get_embedding(),
            "text_node_property": text_property,
            "embedding_node_property": "embedding",
            "index_name": index_name,
            **neo4j_config(),
        }
        if node_label:
            kwargs["node_label"] = node_label
        if retrieval_query:
            kwargs["retrieval_query"] = retrieval_query
        return Neo4jVector(**kwargs)

    def _build_community_store(self) -> Neo4jVector:
        return self._build_store(
            node_label="Community",
            text_property="summary",
            index_name="community_vector_index",
        )

    def _build_lexical_store(self) -> Neo4jVector:
        return self._build_store(
            text_property="text",
            index_name="vector_index",
            retrieval_query="RETURN node.text AS text, score, node {.*, text: Null, embedding:Null} as metadata",
        )

    def get_community_store(self) -> Neo4jVector:
        if self._community_store is None:
            raise HTTPException(500, "Community store not initialized")
        return self._community_store

    def get_lexical_store(self) -> Neo4jVector:
        if self._lexical_store is None:
            raise HTTPException(500, "Lexical store not initialized")
        return self._lexical_store


# Functions for dependency injection
def get_community_store() -> Neo4jVector:
    return VectorStoreService().get_community_store()


def get_lexical_store() -> Neo4jVector:
    return VectorStoreService().get_lexical_store()
