from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from api.enums.vector_store import VectorStoreName
from api.services.vector_store import VectorStoreService

# Global service instance
_vector_store_service: VectorStoreService | None = None


def set_vector_store_service(service: VectorStoreService) -> None:
    """Set the vector store service instance."""
    global _vector_store_service
    _vector_store_service = service


def get_vector_store_service() -> VectorStoreService:
    """Get the vector store service instance."""
    if _vector_store_service is None:
        raise RuntimeError("Vector store service not initialized")
    return _vector_store_service


# Functions for dependency injection
def provide_vector_store(name: VectorStoreName):
    def _get_vector_store() -> Neo4jVector:
        return get_vector_store_service().get_store(name.value)

    return _get_vector_store
