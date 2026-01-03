import json
import logging
import uuid
from copy import deepcopy

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from ingestion.ingestors.base import BaseIngestor
from ingestion.prompts.graph_extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    ONTOLOGY_SYSTEM_PROMPT,
)
from ingestion.schema.extractor import Entity, EntityRelationships, Ontology, Triplet
from ingestion.schema.file import FileMetadata


class PropertyGraphIngestor(BaseIngestor):

    def __init__(
        self,
        description: str,
        llm: BaseChatModel,
        vector_store: Neo4jVector,
        extract_community_summaries: bool = True,
        ontology: Ontology | None = None,
    ):
        self.description = description
        self.llm = llm
        self.vector_store = vector_store
        self.extract_community_summaries = extract_community_summaries
        self.ontology = ontology

        self._ontology_parser = PydanticOutputParser(pydantic_object=Ontology)
        self._triplet_parser = PydanticOutputParser(pydantic_object=EntityRelationships)

        self._ontology_system_prompt = ONTOLOGY_SYSTEM_PROMPT.partial(
            output_format=self._ontology_parser.get_format_instructions()
        )
        self._extraction_system_prompt = EXTRACTION_SYSTEM_PROMPT.partial(
            output_format=self._triplet_parser.get_format_instructions()
        )

    def ingest(self, file_metadata: FileMetadata, documents: list[Document]):
        if not self.ontology:
            self.ontology = self._extract_ontology()

        entities: list[Entity] = []
        triplets: list[Triplet] = []

        entity_storage: dict[str, Entity] = {}

        logging.info(f"Performing NER for {file_metadata['name']}")
        for document in documents:
            current_context_entities = list(entity_storage.values())
            extraction = self._apply_ontology_to_doc(document, current_context_entities)

            for ent in extraction.entities:
                # Update doc_ids for the newly extracted entity
                ent.doc_ids.add(document.id)
                if ent.id in entity_storage:
                    entity_storage[ent.id].properties.update(ent.properties)
                    # Update doc_ids on the existing record
                    entity_storage[ent.id].doc_ids.add(document.id)
                else:
                    entity_storage[ent.id] = ent
            triplets.extend(extraction.triplets)
        logging.info(f"Completed NER for {file_metadata['name']}")

        entities = list(entity_storage.values())

        entities, triplets = self._reassign_entity_ids(entities, triplets)
        logging.info(f"Entites Extracted: {len(entities)}")
        logging.info(f"Triplets Identified: {len(triplets)}")

        node_labels: list[str] = ["Chunk"]
        relationship_labels: list[str] = ["SIMILAR"]
        for entity in entities:
            node_labels.append(entity.entity_label)
            self._create_entity_and_links(entity, file_metadata)
        for triplet in triplets:
            relationship_labels.append(triplet.relationship)
            self._create_triplet_relationship(triplet)

        if self.extract_community_summaries:
            logging.info(f"Extracting Community Summaries for {file_metadata['name']}")
            if not self.llm:
                raise ValueError(
                    "llm must be provided when extract_community_summaries is True. "
                    "Please provide a BaseChatModel instance during initialization."
                )
            self._extract_community_summaries(
                file_metadata, node_labels, relationship_labels
            )
            logging.info(f"Completed Community Summary Extraction")

    def _extract_ontology(self) -> Ontology:
        res = self.llm.invoke(
            [
                SystemMessage(self._ontology_system_prompt.invoke({}).to_string()),
                HumanMessage(self.description),
            ]
        )
        parsed: Ontology = self._ontology_parser.invoke(res.content)
        return parsed

    def _apply_ontology_to_doc(
        self, document: Document, existing_entities: list[Entity]
    ) -> EntityRelationships:

        # Convert entities to dicts while explicitly hiding the doc_ids
        serializable_entities = [
            ent.model_dump(exclude={"doc_ids"}) for ent in existing_entities
        ]

        sytem_prompt = self._extraction_system_prompt.invoke(
            {
                "entity_labels": self.ontology.entity_labels,
                "relationship_rules": self.ontology.relationship_rules,
                "existing_entities": serializable_entities,
            }
        ).to_string()
        res = self.llm.invoke(
            [SystemMessage(sytem_prompt), HumanMessage(document.page_content)]
        )
        parsed: EntityRelationships = self._triplet_parser.invoke(res)
        return parsed

    def _reassign_entity_ids(
        self, entities: list[Entity], triplets: list[Triplet]
    ) -> tuple[list[Entity], list[Triplet]]:

        id_map: dict[str, str] = {}
        new_entities: list[Entity] = []

        for entity in entities:
            new_id = str(uuid.uuid4().hex)
            id_map[entity.id] = new_id

            new_entities.append(
                Entity(
                    id=new_id,
                    entity_label=entity.entity_label,
                    properties=deepcopy(entity.properties),
                    doc_ids=set(entity.doc_ids),
                )
            )

        new_triplets: list[Triplet] = []

        for triplet in triplets:
            if triplet.source_id not in id_map or triplet.target_id not in id_map:
                continue
            new_triplets.append(
                Triplet(
                    source_id=id_map[triplet.source_id],
                    relationship=triplet.relationship,
                    target_id=id_map[triplet.target_id],
                )
            )

        return new_entities, new_triplets

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
