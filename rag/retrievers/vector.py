from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

__all__ = ["vector_search"]


def vector_search(
    query: str, vector_store: VectorStore, top_k: int = 5
) -> list[Document]:
    documents = vector_store.similarity_search(query, top_k)
    return documents


async def avector_search(
    query: str, vector_store: VectorStore, top_k: int = 5
) -> list[Document]:
    documents = vector_store.asimilarity_search(query, top_k)
    return documents
