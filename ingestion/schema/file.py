from typing import TypedDict

from langchain_core.documents import Document
from pydantic import BaseModel


class FileMetadata(TypedDict):
    id: str
    name: str


class File(BaseModel):
    metadata: FileMetadata
    documents: list[Document]
