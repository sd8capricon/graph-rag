import logging

from ingestion.ingestors.base import BaseIngestor
from ingestion.schema.file import File


class Pipeline:
    def __init__(self, ingestor: BaseIngestor):
        self.ingestor = ingestor

    def run(self, files: list[File]):
        logging.info("Started Pipeline for Files")
        for file in files:
            self.ingestor.ingest(file.metadata, file.documents)
        logging.info("Pipeline Completed")
