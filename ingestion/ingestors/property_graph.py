import logging
import uuid
from copy import deepcopy

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from ingestion.extractors.base import BaseExtractor
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
        ontology: Ontology | None = None,
    ):
        self.description = description
        self.llm = llm
        self.vector_store = vector_store
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

        for entity in entities:
            self._create_entity_and_links(entity, file_metadata)
        for triplet in triplets:
            self._create_triplet_relationship(triplet)

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
