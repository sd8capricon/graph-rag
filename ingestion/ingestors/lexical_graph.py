import json
import logging

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from ingestion.extractors.graph_extractor import GraphExtractor
from ingestion.ingestors.base import BaseIngestor
from ingestion.prompts.document_graph import COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT
from ingestion.schema.extractor import Entity, Triplet
from ingestion.schema.file import FileMetadata


class LexicalGraphIngestor(BaseIngestor):

    def __init__(
        self,
        vector_store: Neo4jVector,
        lexical_threshold: float = 0.75,
        extract_community_summaries: bool = True,
        llm: BaseChatModel | None = None,
        graph_extractor: GraphExtractor | None = None,
    ):
        self.vector_store = vector_store
        self.lexical_threshold = lexical_threshold
        self.extract_community_summaries = extract_community_summaries
        self.llm = llm
        self.graph_extractor = graph_extractor

        self._community_summarization_sys_prompt = (
            COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT.invoke({}).to_string()
        )

        self.vector_store.create_new_index()

    def ingest(self, file_metadata: FileMetadata, documents: list[Document]):
        logging.info(f"Ingesting File {file_metadata['name']}")
        document_ids = self._build_vectorstore(documents)

        # List of all node labels and relationships
        node_labels: list[str] = ["Chunk"]
        relationship_labels: list[str] = ["SIMILAR"]

        self._create_file_node(file_metadata)
        self._build_lexical_graph(document_ids)
        if self.graph_extractor:
            logging.info(f"Performing NER for {file_metadata['name']}")
            entities, triplets = self.graph_extractor.extract(documents)
            logging.info(f"Completed NER for {file_metadata['name']}")
            logging.info(f"Entites Extracted: {len(entities)}")
            logging.info(f"Triplets Identified: {len(triplets)}")
            for entity in entities:
                self._create_entity_and_links(entity, file_metadata)
                node_labels.append(entity.entity_label)
            for triplet in triplets:
                self._create_triplet_relationship(triplet)
                relationship_labels.append(triplet.relationship)

            if self.extract_community_summaries:
                logging.info(
                    f"Extracting Community Summaries for {file_metadata['name']}"
                )
                if not self.llm:
                    raise ValueError(
                        "llm must be provided when extract_community_summaries is True. "
                        "Please provide a BaseChatModel instance during initialization."
                    )
                self._extract_community_summaries(
                    file_metadata, node_labels, relationship_labels
                )
                logging.info(f"Completed Community Summary Extraction")

        logging.info(f"Completed Ingesting File {file_metadata['name']}")

    def _build_vectorstore(self, documents: list[Document]) -> list[str]:
        return self.vector_store.add_documents(documents)

    def _create_file_node(self, file_metadata: FileMetadata):
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
                MERGE (from)-[r:SIMILAR]->(to)
                ON CREATE SET r.score = $score
                """,
                params={"from": _from, "to": to, "score": score},
            )

    def _create_entity_and_links(self, entity: Entity, file_metadata: FileMetadata):
        create_entity_query = f"""
        MERGE (e:{entity.entity_label} {{id: $entity_id, source_id: $file_id}})
        SET e += $properties
        """

        try:
            self.vector_store.query(
                create_entity_query,
                params={
                    "entity_id": entity.id,
                    "file_id": file_metadata["id"],
                    "properties": entity.properties or {},
                },
            )
        except Exception as e:
            logging.info("Entity:", entity)
            raise e

        create_doc_rel_query = f"""
        MATCH (e:{entity.entity_label} {{id: $entity_id}})
        MERGE (c:Chunk {{id: $doc_id}})
        MERGE (e)-[:BELONGS_TO]->(c)
        """

        for doc_id in entity.doc_ids:
            self.vector_store.query(
                create_doc_rel_query, params={"entity_id": entity.id, "doc_id": doc_id}
            )

    def _create_triplet_relationship(self, triplet: Triplet):
        query = f"""
        MATCH (s {{id: $source_id}})
        MATCH (t {{id: $target_id}})
        MERGE (s)-[r:{triplet.relationship}]->(t)
        """

        self.vector_store.query(
            query,
            params={"source_id": triplet.source_id, "target_id": triplet.target_id},
        )

    def _extract_community_summaries(
        self,
        file_metadata: FileMetadata,
        node_labels: list[str],
        relationship_labels: list[str],
    ):
        self._make_community_nodes(file_metadata, node_labels, relationship_labels)
        self._generate_community_summaries(file_metadata)

    def _make_community_nodes(
        self,
        file_metadata: FileMetadata,
        node_labels: list[str],
        relationship_labels: list[str],
    ):
        relationship_projections = {
            relationship: {"orientation": "UNDIRECTED"}
            for relationship in relationship_labels
        }

        self.vector_store.query(
            """
            CALL gds.graph.project('kg', $node_labels, $relationship_projections)
            YIELD graphName
            CALL gds.leiden.write("kg", {writeProperty: "community_id"})
            YIELD nodePropertiesWritten
            RETURN nodePropertiesWritten
            """,
            params={
                "node_labels": node_labels,
                "relationship_projections": relationship_projections,
            },
        )

        self.vector_store.query("CALL gds.graph.drop('kg', false) YIELD graphName")

        self.vector_store.query(
            """
            MATCH (e {source_id: $source_id}) 
            WHERE e.community_id IS NOT NULL 
            SET e.community_id = toString(e.source_id) + "_" + toString(e.community_id)
            """,
            params={"source_id": file_metadata["id"]},
        )

        self.vector_store.query(
            """
            MATCH (e {source_id: $source_id})
            WHERE NOT e:Chunk AND e.community_id IS NOT NULL
            WITH DISTINCT e.community_id AS id, collect(e) AS entities
            MERGE (c:Community {id: id})
            ON CREATE SET c.source_id = $source_id
            WITH c, entities
            UNWIND entities AS e
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN c.id;
            """,
            params={"source_id": file_metadata["id"], "node_labels": node_labels},
        )

    def _generate_community_summaries(self, file_metadata: FileMetadata):

        raw_query_result: list[dict[str, str | list[dict]]] = self.vector_store.query(
            """
            MATCH (e)-[:IN_COMMUNITY]->(c {source_id: $source_id})
            WITH c, collect(e) AS entities

            MATCH (src)-[r]->(tgt)
            WHERE src IN entities AND tgt IN entities AND type(r) <> 'IN_COMMUNITY'

            WITH c, src, tgt, r
            ORDER BY src.id, type(r), tgt.id

            WITH c, 
                collect({
                    source_id: {
                        labels: labels(src),
                        properties: apoc.map.clean(properties(src), ['community_id', 'source_id', 'id'], [])
                    },
                    relationship: type(r),
                    target_id: {
                        labels: labels(tgt),
                        properties: apoc.map.clean(properties(src), ['community_id', 'source_id', 'id'], [])
                    }
                }) AS triplets

            RETURN collect({
              id: c.id,
              triplets: triplets
            }) AS result
            """,
            params={"source_id": file_metadata["id"]},
        )

        if not raw_query_result or not raw_query_result[0].get("result"):
            return

        community_data = raw_query_result[0].get("result")
        community_summaries: dict[str, str] = {}

        for mapping in community_data:
            community_id = mapping["id"]
            triplets = mapping["triplets"]

            entities_str = json.dumps(triplets, indent=2)
            try:
                res = self.llm.invoke(
                    [
                        SystemMessage(content=self._community_summarization_sys_prompt),
                        HumanMessage(
                            content=(
                                f"DATASET: The following triplets belong to a single community. "
                                f"Analyze them and provide the summary:\n{entities_str}"
                            )
                        ),
                    ]
                )
                community_summaries[community_id] = res.content
            except Exception as e:
                logging.error(f"Failed to summarize community {community_id}: {e}")

        if community_summaries:
            self.vector_store.query(
                """
                UNWIND $data AS row
                MATCH (c:Community {id: row.cid})
                SET c.summary = row.summary
                """,
                params={
                    "data": [
                        {"cid": cid, "summary": summary}
                        for cid, summary in community_summaries.items()
                    ]
                },
            )
