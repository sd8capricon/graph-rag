from typing import Annotated

from fastapi import File, UploadFile
from pydantic import BaseModel

from common.schema.knowledge_base import KnowledgeBase


class IngestionRequest(BaseModel):
    knowledge_base: KnowledgeBase
    files: Annotated[UploadFile, File()]
