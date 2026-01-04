from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

__all__ = ["vector_search"]


def vector_search(query: str, vector_store: VectorStore) -> list[Document]:
    documents = vector_store.similarity_search(query, 5)
    return documents
