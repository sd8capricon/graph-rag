from pydantic import BaseModel
from common.schema.knowledge_base import KnowledgeBase


class IngestionRequest(BaseModel):
    knowledge_base: KnowledgeBase
    files: list[str]
