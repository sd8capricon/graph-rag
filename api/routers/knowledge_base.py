from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from api.dependencies.knowledge_base import get_kb_service
from api.dependencies.llm import get_llm
from api.dependencies.vector_store import provide_vector_store
from api.enums.vector_store import VectorStoreName
from api.schema.knowledge_base import IngestionRequest
from common.services.knowledge_base import KnowledgeBaseService
from ingestion.ingestors.lexical_graph import LexicalGraphIngestor
from ingestion.ingestors.property_graph import PropertyGraphIngestor
from ingestion.pipeline import Pipeline
from ingestion.readers.markdown import MarkdownReader

router = APIRouter()


@router.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    payload: IngestionRequest,
    knowledge_base_service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
    llm: Annotated[BaseChatModel, Depends(get_llm)],
    lexical_vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.lexical))
    ],
    property_vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.community))
    ],
):
    file_paths = [
        Path(__file__).resolve().parent.parent / "data" / file for file in payload.files
    ]
    files = [MarkdownReader(path).load() for path in file_paths]
    lexical_graph_ingestor = LexicalGraphIngestor(
        vector_store=lexical_vector_store,
    )

    property_graph_ingestor = PropertyGraphIngestor(
        llm=llm,
        vector_store=property_vector_store,
        knowledge_base_service=knowledge_base_service,
    )

    pipeline = Pipeline(
        knowledge_base=payload.knowledge_base,
        knowledge_base_service=knowledge_base_service,
        ingestors=[lexical_graph_ingestor, property_graph_ingestor],
    )

    background_tasks.add_task(pipeline.run, files)
