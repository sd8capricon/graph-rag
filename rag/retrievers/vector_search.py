from langchain_core.vectorstores import VectorStore


def vector_search(query: str, vector_store: VectorStore):
    documents = vector_store.similarity_search(query, 5)
    return documents
