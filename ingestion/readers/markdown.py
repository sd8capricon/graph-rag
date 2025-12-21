import uuid
from pathlib import Path

from langchain_core.documents import Document

from ingestion.readers.base import BaseReader


class MarkdownReader(BaseReader):

    def __init__(self, file_path: Path, chunk_size: int = 512, chunk_overlap: int = 32):
        super().__init__(file_path, chunk_size, chunk_overlap)
        self._read_file()

    def _read_file(self):
        self.file_content = self.file_path.read_text()

    def get_documents(self) -> list[Document]:
        documents = self.text_splitter.create_documents(
            [self.file_content], [{"source": self.file_path.name}]
        )
        for document in documents:
            document.metadata["id"] = uuid.uuid4().hex
        return documents
