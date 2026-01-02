from abc import ABC, abstractmethod


class BaseIngestor(ABC):

    @abstractmethod
    def ingest(self):
        pass
