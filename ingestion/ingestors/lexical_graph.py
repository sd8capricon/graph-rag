import logging

from langchain_core.documents import Document
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from common.schema.knowledge_base import KnowledgeBase
from ingestion.ingestors.base import BaseIngestor
from ingestion.schema.file import FileMetadata


class LexicalGraphIngestor(BaseIngestor):

    def __init__(
        self,
        vector_store: Neo4jVector,
        knowledge_base: KnowledgeBase,
        lexical_threshold: float = 0.75,
    ):
        self.vector_store = vector_store
        self.knowledge_base = knowledge_base
        self.lexical_threshold = lexical_threshold

        self.vector_store.create_new_index()

    def ingest(self, file_metadata: FileMetadata, documents: list[Document]):
        logging.info(f"Ingesting Lexical Graph File {file_metadata['name']}")
        for document in documents:
            document.metadata["knowledge_base_id"] = self.knowledge_base.id
        document_ids = self._build_vectorstore(documents)

        self._create_file_node(file_metadata)
        self._build_lexical_graph(document_ids)

        logging.info(f"Generated Lexical Graph for File {file_metadata['name']}")

    def _build_vectorstore(self, documents: list[Document]) -> list[str]:
        return self.vector_store.add_documents(documents)

    def _create_file_node(self, file_metadata: FileMetadata):
        self.vector_store.query(
            """
                // 1. Create or Update the File node
                MERGE (f:File {id: $id})
                ON CREATE SET 
                    f.name = $name,
                    f.knowledge_base_id = $knowledge_base_id
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
            params={
                "id": file_metadata["id"],
                "name": file_metadata["name"],
                "knowledge_base_id": self.knowledge_base.id,
            },
        )

    def _build_lexical_graph(self, document_ids: list[str]):
        documentid_embeddings = self._get_documents_embedding(document_ids)
        edges = self._get_lexical_edges(documentid_embeddings)
        self._connect_similar_chunks(edges)

    def _get_documents_embedding(self, document_ids: list[str]) -> dict[str, list]:
        embeddings_map: dict[str, list] = {}
        chunks = self.vector_store.query(
            "MATCH (c:Chunk) WHERE c.id in $ids RETURN collect(c {.id, .embedding}) as chunks",
            params={"ids": document_ids},
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
            similarity_search = self.vector_store.similarity_search_with_score_by_vector(
                embedding,
                3,
                query="",  # query is added to avoid internal error, not actually used
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
                MATCH (from:Chunk {id: $from})
                MATCH (to:Chunk {id: $to})
                WHERE from.source_id = to.source_id
                MERGE (from)-[r:SIMILAR]->(to)
                ON CREATE SET r.score = $score
                """,
                params={"from": _from, "to": to, "score": score},
            )
