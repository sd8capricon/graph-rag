import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import ChatOpenAI

from ingestion.extractors.graph_extractor import GraphExtractor
from ingestion.ingestors.graph import DocumentGraphIngestor
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

    file_path = (
        Path(__file__).resolve().parent / "ingestion" / "data" / "alain_prost.md"
    )
    reader = MarkdownReader(file_path)
    file = reader.load()

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", output_dimensionality=768
    )
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        base_url="https://api.groq.com/openai/v1",
        reasoning_effort="medium",
    )

    vector_store = Neo4jVector(
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
    graph_extractor = GraphExtractor(
        description="I have a set of F1 driver resumes. I need to know what information is tracked (like stats and teams), what specific details are inside those categories (like wins or years), and how the drivers, teams, and awards are linked together.",
        llm=llm,
    )
    document_graph_ingestor = DocumentGraphIngestor(
        vector_store=vector_store,
        llm=llm,
        graph_extractor=graph_extractor,
    )

    pipeline = Pipeline(ingestor=document_graph_ingestor)

    # document_graph_ingestor._generate_community_summaries(
    #     file_metadata={"id": "e0dc1bb5b79449ab948633b8a3a183a0"}
    # )

    pipeline.run([file])


if __name__ == "__main__":
    main()
