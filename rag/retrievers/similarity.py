from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from rag.cyphers.chunk import expand_chunk


def similarity_search(query: str, vector_store: VectorStore, top_k: int = 5):
    initial_documents = vector_store.similarity_search(query, top_k)
    doc_ids = [chunk.id for chunk in initial_documents]
    query = vector_store.query(expand_chunk, params={"chunks": doc_ids})
    docs: dict = query[0].get("chunks")
    return docs


async def asimilarity_search(query: str, vector_store: VectorStore, top_k: int = 5):
    initial_documents = await vector_store.asimilarity_search(query, top_k)
    doc_ids = [chunk.id for chunk in initial_documents]
    query = vector_store.query(expand_chunk, params={"chunks": doc_ids})
    docs: dict = query[0].get("chunks")
    return docs
