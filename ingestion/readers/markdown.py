import uuid
from pathlib import Path

from langchain_core.documents import Document

from ingestion.models import FileMetadata
from ingestion.readers.base import BaseReader


class MarkdownReader(BaseReader):

    def __init__(self, file_path: Path, chunk_size: int = 512, chunk_overlap: int = 32):
        super().__init__(file_path, chunk_size, chunk_overlap)
        self._read_file()

    def _read_file(self):
        self.file_name = self.file_path.name
        self.file_id = uuid.uuid4().hex
        self.file_content = self.file_path.read_text()

    def get_file_metadata(self) -> FileMetadata:
        return {"id": self.file_id, "name": self.file_name}

    def get_documents(self) -> list[Document]:
        documents = self.text_splitter.create_documents(
            [self.file_content], [{"source": self.file_name, "source_id": self.file_id}]
        )
        for document in documents:
            document.id = uuid.uuid4().hex
        return documents
