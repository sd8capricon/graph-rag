from contextlib import asynccontextmanager
from api.dependencies.vector_store import VectorStoreService


@asynccontextmanager
async def lifespan():
    # Initialize vector stores on startup
    vector_store_service = VectorStoreService()
    await vector_store_service.initialize()
    yield
