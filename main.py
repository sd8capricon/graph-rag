import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI

from ingestion.ingestors.lexical_graph import LexicalGraphIngestor
from ingestion.ingestors.property_graph import PropertyGraphIngestor
from ingestion.pipeline import Pipeline
from ingestion.readers.markdown import MarkdownReader

load_dotenv()


def setup_logger(level=logging.INFO, fmt="%(asctime)s - %(levelname)s - %(message)s"):
    logger = logging.getLogger()

    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    formatter = logging.Formatter(fmt)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.setLevel(level)


def main() -> None:
    setup_logger()

    file_paths = [
        Path(__file__).resolve().parent / "ingestion" / "data" / "alain_prost.md",
        Path(__file__).resolve().parent / "ingestion" / "data" / "ayrton_senna.md",
    ]

    files = [MarkdownReader(path).load() for path in file_paths]

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", output_dimensionality=768
    )
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b:free",
        base_url="https://openrouter.ai/api/v1",
        reasoning_effort="medium",
    )

    lexical_vector_store = Neo4jVector(
        embedding,
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        text_node_property="text",
        embedding_node_property="embedding",
        index_name="vector_index",
        embedding_dimension=768,
        retrieval_query="RETURN node.text AS text, score, node {.*, text: Null, embedding:Null} as metadata",
    )

    property_vector_store = Neo4jVector(
        embedding,
        username=os.getenv("NEO4J_USER"),
        password=os.getenv("NEO4J_PASSWORD"),
        url=os.getenv("NEO4J_URI"),
        node_label="Community",
        text_node_property="summary",
        embedding_node_property="embedding",
        index_name="community_vector_index",
        embedding_dimension=768,
    )

    lexical_graph_ingestor = LexicalGraphIngestor(
        vector_store=lexical_vector_store,
    )

    property_graph_ingestor = PropertyGraphIngestor(
        description="I have a set of F1 driver resumes. I need to know what information is tracked (like stats and teams), what specific details are inside those categories (like wins or years), and how the drivers, teams, and awards are linked together.",
        llm=llm,
        vector_store=property_vector_store,
    )

    pipeline = Pipeline(ingestors=[lexical_graph_ingestor, property_graph_ingestor])

    # property_graph_ingestor._generate_community_summaries(
    #     file_metadata={"id": "88d4a7e879d54a619cc00ef64f96161f"}
    # )

    pipeline.run(files)


if __name__ == "__main__":
    main()
