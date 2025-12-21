import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from ingestion.readers.markdown import MarkdownReader

load_dotenv()


class Pipeline:

    embedding = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001", output_dimensionality=768
    )

    def __init__(self, documents: list[Document], lexical_threshold: float = 0.75):
        self.vector_store = Neo4jVector(
            self.embedding,
            username=os.getenv("NEO4J_USER"),
            password=os.getenv("NEO4J_PASSWORD"),
            url=os.getenv("NEO4J_URI"),
            text_node_property="text",
            embedding_node_property="embedding",
            index_name="vector_index",
            embedding_dimension=768,
            retrieval_query="RETURN node.text AS text, score, node {.*, text: Null, embedding:Null} as metadata",
        )
        self.vector_store.create_new_index()
        self.documents = documents
        self.lexical_threshold = lexical_threshold

    def run(self):
        self._build_vectorstore()
        self._build_lexical_graph()

    def _build_vectorstore(self):
        self.document_ids = self.vector_store.add_documents(self.documents)

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


if __name__ == "__main__":
    path = Path(__file__).resolve().parent / "data" / "alain_prost.md"
    reader = MarkdownReader(path)
    documents = reader.get_documents()
    pipeline = Pipeline(documents)
    pipeline.run()
