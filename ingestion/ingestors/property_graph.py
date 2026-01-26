import json
import logging
import uuid
from copy import deepcopy

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from common.schema.knowledge_base import KnowledgeBase, Ontology
from common.services.knowledge_base import KnowledgeBaseService
from ingestion.ingestors.base import BaseIngestor
from ingestion.prompts.property_graph import (
    COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    ONTOLOGY_SYSTEM_PROMPT,
)
from ingestion.schema.extractor import Entity, EntityRelationships, Triplet
from ingestion.schema.file import FileMetadata


class PropertyGraphIngestor(BaseIngestor):

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: Neo4jVector,
        extract_community_summaries: bool = True,
        ontology: Ontology | None = None,
        knowledge_base_service: KnowledgeBaseService | None = None,
    ):
        logging.debug("Initializing PropertyGraphIngestor")
        self.llm = llm
        self.vector_store = vector_store
        self.extract_community_summaries = extract_community_summaries
        self.ontology = ontology
        self.knowledge_base_service = knowledge_base_service

        self.knowledge_base: KnowledgeBase | None = None

        self._ontology_parser = PydanticOutputParser(pydantic_object=Ontology)
        self._triplet_parser = PydanticOutputParser(pydantic_object=EntityRelationships)

        self._ontology_system_prompt = ONTOLOGY_SYSTEM_PROMPT.partial(
            output_format=self._ontology_parser.get_format_instructions()
        )
        self._extraction_system_prompt = EXTRACTION_SYSTEM_PROMPT.partial(
            output_format=self._triplet_parser.get_format_instructions()
        )
        self._community_summarization_sys_prompt = (
            COMMUNITY_SUMMARIZATION_SYSTEM_PROMPT.invoke({}).to_string()
        )

        self.vector_store.create_new_index()
        logging.debug("PropertyGraphIngestor initialization complete")

    def ingest(
        self,
        knowledge_base: KnowledgeBase,
        file_metadata: FileMetadata,
        documents: list[Document],
    ):
        logging.debug(
            f"Starting ingestion for file: {file_metadata['name']}, documents count: {len(documents)}"
        )
        if not self.ontology:
            logging.debug("Ontology not found, extracting from knowledge base")
            if not knowledge_base.ontology:
                logging.debug("Extracting ontology from knowledge extraction prompt")
                knowledge_base.ontology = self._extract_ontology(
                    knowledge_base.knowledge_extraction_prompt
                )
                logging.debug(
                    f"Ontology extracted with {len(knowledge_base.ontology.entity_labels)} entity labels"
                )
                self.knowledge_base_service.upsert(knowledge_base)
                logging.debug("Ontology saved to knowledge base")
            self.ontology = knowledge_base.ontology

        if not self.knowledge_base:
            self.knowledge_base = knowledge_base

        entities: list[Entity] = []
        triplets: list[Triplet] = []

        entity_storage: dict[str, Entity] = {}

        logging.debug(f"Performing NER for {file_metadata['name']}")
        for idx, document in enumerate(documents):
            logging.debug(
                f"Processing document {idx + 1}/{len(documents)}: {document.id}"
            )
            current_context_entities = list(entity_storage.values())
            logging.debug(
                f"Current context has {len(current_context_entities)} entities"
            )
            extraction = self._apply_ontology_to_doc(document, current_context_entities)
            logging.debug(
                f"Extracted {len(extraction.entities)} entities and {len(extraction.triplets)} triplets from document"
            )

            for ent in extraction.entities:
                # Update doc_ids for the newly extracted entity
                ent.doc_ids.add(document.id)
                if ent.id in entity_storage:
                    logging.debug(f"Updating existing entity: {ent.id}")
                    entity_storage[ent.id].properties.update(ent.properties)
                    # Update doc_ids on the existing record
                    entity_storage[ent.id].doc_ids.add(document.id)
                else:
                    logging.debug(f"Adding new entity: {ent.id} ({ent.entity_label})")
                    entity_storage[ent.id] = ent
            triplets.extend(extraction.triplets)
        logging.debug(f"Completed NER for {file_metadata['name']}")

        entities = list(entity_storage.values())
        logging.debug(f"Total unique entities before ID reassignment: {len(entities)}")

        logging.debug("Reassigning entity IDs")
        entities, triplets = self._reassign_entity_ids(entities, triplets)
        logging.debug(f"Entities Extracted: {len(entities)}")
        logging.debug(f"Triplets Identified: {len(triplets)}")

        node_labels: list[str] = ["Chunk"]
        relationship_labels: list[str] = ["SIMILAR"]
        logging.debug("Creating entity nodes and relationships")
        for idx, entity in enumerate(entities):
            logging.debug(
                f"Creating entity {idx + 1}/{len(entities)}: {entity.id} ({entity.entity_label})"
            )
            node_labels.append(entity.entity_label)
            self._create_entity_and_links(entity, file_metadata)
        logging.debug(f"Created {len(entities)} entity nodes")

        logging.debug("Creating triplet relationships")
        for idx, triplet in enumerate(triplets):
            logging.debug(
                f"Creating triplet {idx + 1}/{len(triplets)}: {triplet.source_id} -[{triplet.relationship}]-> {triplet.target_id}"
            )
            relationship_labels.append(triplet.relationship)
            self._create_triplet_relationship(triplet)
        logging.debug(f"Created {len(triplets)} relationships")

        if self.extract_community_summaries:
            logging.debug(f"Extracting Community Summaries for {file_metadata['name']}")
            if not self.llm:
                raise ValueError(
                    "llm must be provided when extract_community_summaries is True. "
                    "Please provide a BaseChatModel instance during initialization."
                )
            logging.debug(
                f"Community summary extraction enabled, processing with node_labels={len(node_labels)}, relationship_labels={len(relationship_labels)}"
            )
            self._extract_community_summaries(
                file_metadata, node_labels, relationship_labels
            )
            logging.debug(f"Completed Community Summary Extraction")

        logging.debug(f"Ingestion complete for file: {file_metadata['name']}")

    def _extract_ontology(self, knowledge_extraction_prompt: str) -> Ontology:
        logging.debug("Invoking LLM to extract ontology")
        res = self.llm.invoke(
            [
                SystemMessage(self._ontology_system_prompt.invoke({}).to_string()),
                HumanMessage(knowledge_extraction_prompt),
            ]
        )
        logging.debug("Parsing ontology response")
        parsed: Ontology = self._ontology_parser.invoke(res.content)
        logging.debug(
            f"Ontology parsed successfully: {len(parsed.entity_labels)} entity labels, {len(parsed.relationship_rules)} relationship rules"
        )
        return parsed

    def _apply_ontology_to_doc(
        self, document: Document, existing_entities: list[Entity]
    ) -> EntityRelationships:
        logging.debug(
            f"Applying ontology to document: {document.id}, existing entities: {len(existing_entities)}"
        )

        # Convert entities to dicts while explicitly hiding the doc_ids
        serializable_entities = [
            ent.model_dump(exclude={"doc_ids"}) for ent in existing_entities
        ]
        logging.debug(
            f"Prepared {len(serializable_entities)} entities for serialization"
        )

        logging.debug("Building extraction prompt")
        sytem_prompt = self._extraction_system_prompt.invoke(
            {
                "entity_labels": self.ontology.entity_labels,
                "relationship_rules": self.ontology.relationship_rules,
                "existing_entities": serializable_entities,
            }
        ).to_string()
        logging.debug("Invoking LLM for entity and relationship extraction")
        res = self.llm.invoke(
            [SystemMessage(sytem_prompt), HumanMessage(document.page_content)]
        )
        logging.debug("Parsing entity and relationship extraction response")
        parsed: EntityRelationships = self._triplet_parser.invoke(res)
        logging.debug(
            f"Extraction complete: {len(parsed.entities)} entities, {len(parsed.triplets)} relationships"
        )
        return parsed

    def _reassign_entity_ids(
        self, entities: list[Entity], triplets: list[Triplet]
    ) -> tuple[list[Entity], list[Triplet]]:
        logging.debug(
            f"Reassigning IDs for {len(entities)} entities and {len(triplets)} triplets"
        )

        id_map: dict[str, str] = {}
        new_entities: list[Entity] = []

        for idx, entity in enumerate(entities):
            new_id = str(uuid.uuid4().hex)
            id_map[entity.id] = new_id
            logging.debug(
                f"Mapped entity {idx + 1}/{len(entities)}: {entity.id} -> {new_id}"
            )

            new_entities.append(
                Entity(
                    id=new_id,
                    entity_label=entity.entity_label,
                    properties=deepcopy(entity.properties),
                    doc_ids=set(entity.doc_ids),
                )
            )

        logging.debug(
            f"Created {len(new_entities)} new entity objects with reassigned IDs"
        )
        new_triplets: list[Triplet] = []

        for idx, triplet in enumerate(triplets):
            if triplet.source_id not in id_map or triplet.target_id not in id_map:
                logging.debug(
                    f"Skipping triplet {idx + 1}/{len(triplets)}: missing source or target ID mapping"
                )
                continue
            logging.debug(
                f"Mapping triplet {idx + 1}/{len(triplets)}: {id_map[triplet.source_id]} -[{triplet.relationship}]-> {id_map[triplet.target_id]}"
            )
            new_triplets.append(
                Triplet(
                    source_id=id_map[triplet.source_id],
                    relationship=triplet.relationship,
                    target_id=id_map[triplet.target_id],
                )
            )

        logging.debug(
            f"Reassignment complete: {len(new_entities)} entities and {len(new_triplets)} triplets mapped"
        )
        return new_entities, new_triplets

    def _create_entity_and_links(self, entity: Entity, file_metadata: FileMetadata):
        logging.debug(f"Creating entity node: {entity.id} ({entity.entity_label})")
        create_entity_query = f"""
        MERGE (e:{entity.entity_label} {{id: $entity_id, knowledge_base_id: $knowledge_base_id, source_id: $file_id}})
        SET e += $properties
        """

        try:
            logging.debug(
                f"Executing MERGE query for entity {entity.id} with properties: {entity.properties}"
            )
            self.vector_store.query(
                create_entity_query,
                params={
                    "entity_id": entity.id,
                    "knowledge_base_id": self.knowledge_base.id,
                    "file_id": file_metadata["id"],
                    "properties": entity.properties or {},
                },
            )
            logging.debug(f"Entity node created successfully: {entity.id}")
        except Exception as e:
            logging.error(f"Failed to create entity node: {entity.id}, error: {str(e)}")
            logging.debug("Entity:", entity)
            raise e

        create_doc_rel_query = f"""
        MATCH (e:{entity.entity_label} {{id: $entity_id}})
        MERGE (c:Chunk {{id: $doc_id}})
        MERGE (e)-[:BELONGS_TO]->(c)
        """

        logging.debug(
            f"Creating BELONGS_TO relationships for entity {entity.id} with {len(entity.doc_ids)} documents"
        )
        for doc_idx, doc_id in enumerate(entity.doc_ids):
            logging.debug(
                f"Creating BELONGS_TO relationship {doc_idx + 1}/{len(entity.doc_ids)}: {entity.id} -> {doc_id}"
            )
            self.vector_store.query(
                create_doc_rel_query, params={"entity_id": entity.id, "doc_id": doc_id}
            )

    def _create_triplet_relationship(self, triplet: Triplet):
        logging.debug(
            f"Creating triplet relationship: {triplet.source_id} -[{triplet.relationship}]-> {triplet.target_id}"
        )
        query = f"""
        MATCH (s {{id: $source_id}})
        MATCH (t {{id: $target_id}})
        MERGE (s)-[r:{triplet.relationship}]->(t)
        """

        try:
            self.vector_store.query(
                query,
                params={"source_id": triplet.source_id, "target_id": triplet.target_id},
            )
            logging.debug(f"Triplet relationship created successfully")
        except Exception as e:
            logging.error(f"Failed to create triplet relationship: {str(e)}")
            raise e

    def _extract_community_summaries(
        self,
        file_metadata: FileMetadata,
        node_labels: list[str],
        relationship_labels: list[str],
    ):
        logging.debug(f"Starting community summary extraction")
        logging.debug(
            f"Creating community nodes with {len(node_labels)} node labels and {len(relationship_labels)} relationship labels"
        )
        self._make_community_nodes(file_metadata, node_labels, relationship_labels)
        logging.debug(f"Community nodes created, now generating summaries")
        self._generate_community_summaries(file_metadata)
        logging.debug(f"Community summary extraction complete")

    def _make_community_nodes(
        self,
        file_metadata: FileMetadata,
        node_labels: list[str],
        relationship_labels: list[str],
    ):
        logging.debug(f"Creating community nodes for file: {file_metadata['name']}")
        logging.debug(
            f"Building relationship projections from {len(relationship_labels)} labels"
        )
        relationship_projections = {
            relationship: {"orientation": "UNDIRECTED"}
            for relationship in relationship_labels
        }
        logging.debug(
            f"Created {len(relationship_projections)} relationship projections"
        )

        logging.debug("Projecting graph for community detection")
        self.vector_store.query(
            """//cypher
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
        logging.debug("Graph projection and community detection completed")

        logging.debug("Dropping temporary graph projection")
        self.vector_store.query("CALL gds.graph.drop('kg', false) YIELD graphName")
        logging.debug("Graph projection dropped")

        logging.debug("Updating community IDs to be globally unique")
        self.vector_store.query(
            """//cypher
            MATCH (e {source_id: $source_id}) 
            WHERE e.community_id IS NOT NULL 
            SET e.community_id = toString(e.source_id) + "_" + toString(e.community_id)
            """,
            params={"source_id": file_metadata["id"]},
        )
        logging.debug("Community IDs updated")

        logging.debug("Creating Community nodes and IN_COMMUNITY relationships")
        self.vector_store.query(
            """//cypher
            MATCH (e {source_id: $source_id})
            WHERE NOT e:Chunk AND e.community_id IS NOT NULL
            WITH DISTINCT e.community_id AS id, collect(e) AS entities
            MERGE (c:Community {id: id})
            ON CREATE SET c.source_id = $source_id, c.knowledge_base_id = $knowledge_base_id
            WITH c, entities
            UNWIND entities AS e
            MERGE (e)-[:IN_COMMUNITY]->(c)
            RETURN c.id;
            """,
            params={
                "source_id": file_metadata["id"],
                "knowledge_base_id": self.knowledge_base.id,
                "node_labels": node_labels,
            },
        )
        logging.debug("Community nodes and relationships created")

    def _generate_community_summaries(self, file_metadata: FileMetadata):
        logging.debug(
            f"Generating summaries for communities in file: {file_metadata['name']}"
        )
        logging.debug("Querying community data")
        embedding = self.vector_store.embedding
        raw_query_result: list[dict[str, str | list[dict]]] = self.vector_store.query(
            """//cypher
            MATCH (e)-[:IN_COMMUNITY]->(c {source_id: $source_id})
            WITH c, collect(e) AS entities

            MATCH (src)-[r]->(tgt)
            WHERE src IN entities AND tgt IN entities AND type(r) <> 'IN_COMMUNITY'

            WITH c, src, tgt, r
            ORDER BY src.id, type(r), tgt.id

            WITH c, 
                collect({
                    source: {
                        labels: labels(src),
                        properties: apoc.map.clean(properties(src), ['community_id', 'source_id', 'id'], [])
                    },
                    relationship: type(r),
                    target: {
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
            logging.debug("No community data found to summarize")
            return

        community_data = raw_query_result[0].get("result")
        logging.debug(f"Retrieved data for {len(community_data)} communities")
        community_summaries: dict[str, dict] = {}

        for idx, mapping in enumerate(community_data):
            community_id = mapping["id"]
            triplets = mapping["triplets"]
            logging.debug(
                f"Processing community {idx + 1}/{len(community_data)}: {community_id} with {len(triplets)} triplets"
            )

            entities_str = json.dumps(triplets, indent=2)
            try:
                logging.debug(f"Invoking LLM to summarize community {community_id}")
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
                community_summaries[community_id] = {"summary": res.content}
                logging.debug(
                    f"Successfully generated summary for community {community_id}"
                )
            except Exception as e:
                logging.error(f"Failed to summarize community {community_id}: {e}")

        if not community_summaries:
            logging.debug("No community summaries generated")
            return

        logging.debug(
            f"Generated {len(community_summaries)} community summaries, now generating embeddings"
        )
        # Generate vectors for community summaries (order-safe)
        community_ids = list(community_summaries.keys())
        summaries = [community_summaries[cid]["summary"] for cid in community_ids]
        logging.debug(f"Embedding {len(summaries)} community summaries")
        embeddings = embedding.embed_documents(summaries)
        logging.debug(f"Generated {len(embeddings)} embeddings")
        for idx, cid in enumerate(community_ids):
            community_summaries[cid]["embedding"] = embeddings[idx]
            logging.debug(
                f"Assigned embedding {idx + 1}/{len(community_ids)} to community {cid}"
            )

        if community_summaries:
            logging.debug(
                f"Saving {len(community_summaries)} community summaries and embeddings to database"
            )
            self.vector_store.query(
                """
                UNWIND $data AS row
                MATCH (c:Community {id: row.cid})
                SET c.summary = row.summary, c.embedding = row.embedding
                """,
                params={
                    "data": [
                        {
                            "cid": cid,
                            "summary": data["summary"],
                            "embedding": data["embedding"],
                        }
                        for cid, data in community_summaries.items()
                    ]
                },
            )
            logging.debug(
                f"Successfully saved {len(community_summaries)} community summaries to database"
            )
