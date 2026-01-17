from dataclasses import dataclass

from rag.schema.retrievers import DriftConfig


@dataclass
class RAGContext:
    drift_config: DriftConfig
