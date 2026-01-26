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
        lexical_threshold: float = 0.75,
    ):
        logging.debug("Initializing LexicalGraphIngestor")
        self.vector_store = vector_store
        self.lexical_threshold = lexical_threshold
        logging.debug(f"Lexical threshold set to: {self.lexical_threshold}")

        self.vector_store.create_new_index()
        logging.debug("Vector store index created")

    def ingest(
        self,
        knowledge_base: KnowledgeBase,
        file_metadata: FileMetadata,
        documents: list[Document],
    ):
        logging.info(f"Ingesting Lexical Graph File {file_metadata['name']}")
        logging.debug(
            f"Processing {len(documents)} documents for file: {file_metadata['name']}"
        )

        logging.debug("Adding knowledge_base_id to document metadata")
        for idx, document in enumerate(documents):
            document.metadata["knowledge_base_id"] = knowledge_base.id
            logging.debug(
                f"Updated metadata for document {idx + 1}/{len(documents)}: {document.id}"
            )

        logging.debug("Building vector store with documents")
        document_ids = self._build_vectorstore(documents)
        logging.debug(f"Vector store built with {len(document_ids)} documents")

        logging.debug("Creating file node")
        self._create_file_node(knowledge_base, file_metadata)
        logging.debug("File node created")

        logging.debug("Building lexical graph")
        self._build_lexical_graph(document_ids)
        logging.debug("Lexical graph built")

        logging.info(f"Generated Lexical Graph for File {file_metadata['name']}")

    def _build_vectorstore(self, documents: list[Document]) -> list[str]:
        logging.debug(f"Adding {len(documents)} documents to vector store")
        document_ids = self.vector_store.add_documents(documents)
        logging.debug(
            f"Successfully added {len(document_ids)} documents to vector store"
        )
        return document_ids

    def _create_file_node(
        self, knowledge_base: KnowledgeBase, file_metadata: FileMetadata
    ):
        logging.debug(
            f"Creating file node: {file_metadata['id']} ({file_metadata['name']})"
        )
        try:
            result = self.vector_store.query(
                """
                    // 1. Create or Update the File node
                    MERGE (f:File {id: $id})
                    ON CREATE SET 
                        f.name = $name,
                        f.knowledge_base_id = $knowledge_base_id,
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
                    "knowledge_base_id": knowledge_base.id,
                },
            )
            logging.debug(f"File node created successfully: {file_metadata['id']}")
        except Exception as e:
            logging.error(
                f"Failed to create file node for {file_metadata['id']}: {str(e)}"
            )
            raise

    def _build_lexical_graph(self, document_ids: list[str]):
        logging.debug(f"Building lexical graph for {len(document_ids)} documents")

        logging.debug("Retrieving document embeddings")
        documentid_embeddings = self._get_documents_embedding(document_ids)
        logging.debug(
            f"Retrieved embeddings for {len(documentid_embeddings)} documents"
        )

        logging.debug("Computing lexical edges")
        edges = self._get_lexical_edges(documentid_embeddings)
        logging.debug(
            f"Found {len(edges)} similar document pairs above threshold ({self.lexical_threshold})"
        )

        logging.debug(f"Connecting {len(edges)} similar chunks")
        self._connect_similar_chunks(edges)
        logging.debug("Lexical graph construction complete")

    def _get_documents_embedding(self, document_ids: list[str]) -> dict[str, list]:
        logging.debug(f"Fetching embeddings for {len(document_ids)} documents")
        embeddings_map: dict[str, list] = {}
        try:
            chunks = self.vector_store.query(
                "MATCH (c:Chunk) WHERE c.id in $ids RETURN collect(c {.id, .embedding}) as chunks",
                params={"ids": document_ids},
            )[0]["chunks"]
            logging.debug(f"Retrieved {len(chunks)} chunks from database")

            for idx, chunk in enumerate(chunks):
                vector = chunk["embedding"]
                doc_id = chunk["id"]
                embeddings_map[doc_id] = vector
                logging.debug(f"Processed embedding {idx + 1}/{len(chunks)}: {doc_id}")

            logging.debug(
                f"Successfully created embeddings map with {len(embeddings_map)} entries"
            )
        except Exception as e:
            logging.error(f"Failed to fetch embeddings: {str(e)}")
            raise
        return embeddings_map

    def _get_lexical_edges(
        self, id_embedding_map: dict[str, list]
    ) -> list[tuple[str, str, float]]:
        logging.debug(f"Computing lexical edges for {len(id_embedding_map)} documents")
        edges: list[tuple[str, str, float]] = []

        for idx, (doc_id, embedding) in enumerate(id_embedding_map.items()):
            logging.debug(
                f"Searching for similar documents {idx + 1}/{len(id_embedding_map)}: {doc_id}"
            )
            similarity_search = self.vector_store.similarity_search_with_score_by_vector(
                embedding,
                3,
                query="",  # query is added to avoid internal error, not actually used
            )
            logging.debug(
                f"Found {len(similarity_search)} potential matches for {doc_id}"
            )

            matches_above_threshold = 0
            for similar_document, score in similarity_search:
                similar_doc_id = similar_document.metadata["id"]
                if doc_id == similar_doc_id:
                    logging.debug(f"Skipping self-match: {doc_id}")
                    continue
                if score > self.lexical_threshold:
                    logging.debug(
                        f"Match above threshold: {doc_id} -> {similar_doc_id} (score: {score:.4f})"
                    )
                    edges.append((doc_id, similar_doc_id, score))
                    matches_above_threshold += 1
                else:
                    logging.debug(
                        f"Match below threshold: {doc_id} -> {similar_doc_id} (score: {score:.4f})"
                    )

            logging.debug(
                f"Document {doc_id}: {matches_above_threshold} matches above threshold"
            )

        logging.debug(f"Total lexical edges computed: {len(edges)}")
        return edges

    def _connect_similar_chunks(self, edges: list[tuple[str, str, float]]):
        logging.debug(f"Connecting {len(edges)} similar chunk pairs")

        for idx, (_from, to, score) in enumerate(edges):
            logging.debug(
                f"Creating SIMILAR relationship {idx + 1}/{len(edges)}: {_from} -> {to} (score: {score:.4f})"
            )
            try:
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
                logging.debug(
                    f"Successfully created SIMILAR relationship: {_from} -> {to}"
                )
            except Exception as e:
                logging.error(
                    f"Failed to create SIMILAR relationship {_from} -> {to}: {str(e)}"
                )
                raise

        logging.debug(f"All {len(edges)} SIMILAR relationships created successfully")
