import logging

from common.schema.knowledge_base import KnowledgeBase
from common.services.knowledge_base import KnowledgeBaseService
from ingestion.ingestors.base import BaseIngestor
from ingestion.schema.file import File


class Pipeline:
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        knowledge_base_service: KnowledgeBaseService,
        ingestors: list[BaseIngestor],
    ):
        self.knowledge_base = knowledge_base
        self.knowledge_base_service = knowledge_base_service
        self.ingestors = ingestors

    def run(self, files: list[File]):
        logging.info("Started Pipeline for Files")
        self.knowledge_base_service.upsert(self.knowledge_base)
        for file in files:
            for ingestor in self.ingestors:
                ingestor.ingest(file.metadata, file.documents)
        logging.info("Pipeline Completed")
