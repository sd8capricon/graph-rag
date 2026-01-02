from schema.file import File

from ingestion.ingestors.base import BaseIngestor


class Pipeline:
    def __init__(self, ingestor: BaseIngestor):
        self.ingestor = ingestor

    def run(self, files: list[File]):
        for file in files:
            self.ingestor.ingest(file.metadata, file.documents)
