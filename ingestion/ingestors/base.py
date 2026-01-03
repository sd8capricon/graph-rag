from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from ingestion.schema.file import FileMetadata


class BaseIngestor(ABC):

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    @abstractmethod
    def ingest(self, file_metadata: FileMetadata, documents: list[Document]):
        pass
