from abc import ABC, abstractmethod
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.models import FileMetadata


class BaseReader(ABC):

    def __init__(self, file_path: Path, chunk_size: int = 512, chunk_overlap: int = 32):
        super().__init__()

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.file_path = file_path

    @abstractmethod
    def _read_file(self):
        pass

    @abstractmethod
    def get_file_metadata(self) -> FileMetadata:
        pass

    @abstractmethod
    def get_documents(self) -> list[Document]:
        pass
