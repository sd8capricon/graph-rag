from schema.file import File

from ingestion.ingestors.base import BaseIngestor


class Pipeline:
    def __init__(self, ingestor: BaseIngestor):
        self.ingestor = ingestor

    def run(self, files: list[File]):
        for file in files:
            self.ingestor.ingest(file.metadata, file.documents)


if __name__ == "__main__":
    import os

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector
    from langchain_openai import ChatOpenAI

    from ingestion.extractors.graph_extractor import GraphExtractor
    from ingestion.ingestors.graph import DocumentGraphIngestor

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", output_dimensionality=768
    )
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
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
    graph_extractor = GraphExtractor()

    pipeline = Pipeline(
        ingestor=DocumentGraphIngestor(
            vector_store=vector_store,
            graph_extractor=graph_extractor,
        )
    )
