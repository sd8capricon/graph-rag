import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from ingestion.extractors.graph_extractor import GraphExtractor
from ingestion.readers.markdown import MarkdownReader
from ingestion.schema.extractor import Entity, Triplet
from ingestion.schema.file import FileMetadata

load_dotenv()


class DocumentGraphIngestor:

    def __init__(
        self,
        vector_store: Neo4jVector,
        file_metadata: FileMetadata,
        documents: list[Document],
        lexical_threshold: float = 0.75,
        graph_extractor: GraphExtractor | None = None,
    ):
        self.vector_store = vector_store
        self.file_metadata = file_metadata
        self.documents = documents
        self.lexical_threshold = lexical_threshold
        self.graph_extractor = graph_extractor

        self.vector_store.create_new_index()

    def run(self):
        self._build_vectorstore()
        self._create_file_node()
        self._build_lexical_graph()
        if self.graph_extractor:
            entities, triplets = self.graph_extractor.extract(self.documents)
            for entity in entities:
                self._create_entity_and_links(entity)
            for triplet in triplets:
                self._create_triplet_relationship(triplet)

    def _build_vectorstore(self):
        self.document_ids = self.vector_store.add_documents(self.documents)

    def _create_file_node(self):
        self.vector_store.query(
            """
                // 1. Create or Update the File node
                MERGE (f:File {id: $id})
                ON CREATE SET 
                    f.name = $name,
                    f.createdAt = timestamp()
                ON MATCH SET 
                    f.name = $name
                // 2. Find all existing Chunks that belong to this file
                WITH f
                MATCH (c:Chunk {source_id: f.id})
                // 3. Create the relationship
                MERGE (c)-[:CHUNK_OF]->(f)
                RETURN f
            """,
            params={"id": file_metadata["id"], "name": file_metadata["name"]},
        )

    def _build_lexical_graph(self):
        documentid_embeddings = self._get_documents_embedding()
        edges = self._get_lexical_edges(documentid_embeddings)
        self._connect_similar_chunks(edges)

    def _get_documents_embedding(self) -> dict[str, list]:
        embeddings_map: dict[str, list] = {}
        chunks = self.vector_store.query(
            "MATCH (c:Chunk) WHERE c.id in $ids RETURN collect(c {.id, .embedding}) as chunks",
            params={"ids": self.document_ids},
        )[0]["chunks"]
        for chunk in chunks:
            vector = chunk["embedding"]
            doc_id = chunk["id"]
            embeddings_map[doc_id] = vector
        return embeddings_map

    def _get_lexical_edges(
        self, id_embedding_map: dict[str, list]
    ) -> list[tuple[str, str, float]]:
        edges: list[tuple[str, str, float]] = []
        for doc_id, embedding in id_embedding_map.items():
            similarity_search = (
                self.vector_store.similarity_search_with_score_by_vector(
                    embedding, 3, query=""
                )
            )
            for similar_document, score in similarity_search:
                similar_doc_id = similar_document.metadata["id"]
                if doc_id == similar_doc_id:
                    continue
                if score > self.lexical_threshold:
                    edges.append((doc_id, similar_doc_id, score))
        return edges

    def _connect_similar_chunks(self, edges: list[tuple[str, str, float]]):
        for _from, to, score in edges:
            self.vector_store.query(
                """
                MATCH (from:Chunk {id: $from}), (to:Chunk {id: $to})
                MERGE (from)-[r:SIMILAR]->(to)
                ON CREATE SET r.score = $score
                """,
                params={"from": _from, "to": to, "score": score},
            )

    def _create_entity_and_links(self, entity: Entity):
        create_entity_query = f"""
        MERGE (e:{entity.entity_label} {{id: $entity_id}})
        SET e += $properties
        """
        self.vector_store.query(
            create_entity_query,
            {
                "entity_id": entity.id,
                "properties": entity.properties or {},
            },
        )

        create_doc_rel_query = f"""
        MATCH (e:{entity.entity_label} {{id: $entity_id}})
        MERGE (d:Document {{id: $doc_id}})
        MERGE (e)-[:BELONGS_TO]->(d)
        """

        for doc_id in entity.doc_ids:
            self.vector_store.query(
                create_doc_rel_query, {"entity_id": entity.id, "doc_id": doc_id}
            )

    def _create_triplet_relationship(self, triplet: Triplet):
        query = f"""
        MATCH (s {{id: $source_id}})
        MATCH (t {{id: $target_id}})
        MERGE (s)-[r:{triplet.relationship}]->(t)
        """

        self.vector_store.query(
            query, {"source_id": triplet.source_id, "target_id": triplet.target_id}
        )


if __name__ == "__main__":
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_openai import ChatOpenAI

    path = Path(__file__).resolve().parent / "data" / "alain_prost.md"
    reader = MarkdownReader(path)
    file_metadata = reader.get_file_metadata()
    documents = reader.get_documents()

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

    graph_exractor = GraphExtractor(
        description="I have a set of F1 driver resumes. I need to know what information is tracked (like stats and teams), what specific details are inside those categories (like wins or years), and how the drivers, teams, and awards are linked together.",
        llm=llm,
    )

    pipeline = DocumentGraphIngestor(
        vector_store=vector_store,
        file_metadata=file_metadata,
        documents=documents,
        graph_extractor=graph_exractor,
    )
    pipeline.run()
