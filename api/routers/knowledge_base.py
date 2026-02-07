import shutil
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Form
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

API_ROOT = Path(__file__).resolve().parent.parent
TEMP_ROOT = API_ROOT / "temp"
TEMP_ROOT.mkdir(exist_ok=True)

router = APIRouter()


@router.post("/ingest")
async def ingest(
    background_tasks: BackgroundTasks,
    payload: Annotated[IngestionRequest, Form()],
    knowledge_base_service: Annotated[KnowledgeBaseService, Depends(get_kb_service)],
    llm: Annotated[BaseChatModel, Depends(get_llm)],
    lexical_vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.lexical))
    ],
    property_vector_store: Annotated[
        Neo4jVector, Depends(provide_vector_store(VectorStoreName.community))
    ],
):
    temp_dir = TEMP_ROOT / payload.knowledge_base.id
    temp_dir.mkdir(exist_ok=True)
    extract_dir = temp_dir / "extracted"
    temp_dir.mkdir(parents=True)
    extract_dir.mkdir()

    # Save ZIP
    zip_path = temp_dir / payload.files.filename
    with zip_path.open("wb") as f:
        shutil.copyfileobj(payload.files.file, f)

    # Extract ZIP
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    # Collect extracted files (e.g. markdown only)
    file_paths = [
        p for p in extract_dir.rglob("*") if p.is_file() and p.suffix in {".md", ".txt"}
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
