from pydantic import BaseModel

from api.enums.vector_store import VectorStoreName


class VectorStoreConfig(BaseModel):
    """Configuration for a vector store."""

    name: VectorStoreName
    node_label: str | None = None
    text_property: str = "text"
    index_name: str = "vector_index"
    retrieval_query: str | None = None
