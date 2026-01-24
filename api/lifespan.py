from contextlib import asynccontextmanager

from api.dependencies.vector_store import set_vector_store_service
from api.enums.vector_store import VectorStoreName
from api.schema.vector_store import VectorStoreConfig
from api.services.vector_store import VectorStoreService


@asynccontextmanager
async def lifespan(app):
    # Initialize vector stores on startup
    service = VectorStoreService()
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
    await service.initialize(configs)
    set_vector_store_service(service)
    yield
