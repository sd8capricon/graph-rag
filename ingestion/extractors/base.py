from abc import ABC, abstractmethod

from langchain_core.documents import Document


class BaseExtractor(ABC):

    @abstractmethod
    def __init__(self, documents: list[Document]):
        pass

    @abstractmethod
    def extract(self):
        pass
