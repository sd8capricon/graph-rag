from abc import ABC, abstractmethod

from langchain_core.documents import Document

from ingestion.schema.file import FileMetadata


class BaseIngestor(ABC):

    @abstractmethod
    def ingest(self, file_metadata: FileMetadata, documents: list[Document]):
        pass
