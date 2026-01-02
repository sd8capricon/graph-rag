import uuid
from copy import deepcopy

from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser

from ingestion.extractors.base import BaseExtractor
from ingestion.prompts.graph_extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    ONTOLOGY_SYSTEM_PROMPT,
)
from ingestion.schema.extractor import Entity, EntityRelationships, Ontology, Triplet


class GraphExtractor(BaseExtractor):
    def __init__(
        self,
        description: str,
        llm: BaseChatModel,
        ontology: Ontology = None,
    ):
        self.description = description
        self.llm = llm
        self.ontology = ontology
        self._ontology_parser = PydanticOutputParser(pydantic_object=Ontology)
        self._triplet_parser = PydanticOutputParser(pydantic_object=EntityRelationships)

        self._ontology_system_prompt = ONTOLOGY_SYSTEM_PROMPT.partial(
            output_format=self._ontology_parser.get_format_instructions()
        )
        self._extraction_system_prompt = EXTRACTION_SYSTEM_PROMPT.partial(
            output_format=self._triplet_parser.get_format_instructions()
        )

        self.entities: list[Entity] = []
        self.triplets: list[Triplet] = []

    def extract(self, documents: list[Document]) -> tuple[list[Entity], list[Triplet]]:
        if not self.ontology:
            self.ontology = self._extract_ontology()

        entity_storage: dict[str, Entity] = {}
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
            self.triplets.extend(extraction.triplets)

        self.entities = list(entity_storage.values())

        return self._reassign_entity_ids()

    def _extract_ontology(self) -> Ontology:
        # system_prompt.partial
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

    def _reassign_entity_ids(self) -> tuple[list[Entity], list[Triplet]]:

        id_map: dict[str, str] = {}
        new_entities: list[Entity] = []

        for entity in self.entities:
            new_id = str(uuid.uuid4())
            id_map[new_id] = new_id

            new_entities.append(
                Entity(
                    id=new_id,
                    entity_label=entity.entity_label,
                    properties=deepcopy(entity.properties),
                    doc_ids=set(entity.doc_ids),
                )
            )

        new_triplets: list[Triplet] = []

        for triplet in self.triplets:
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


if __name__ == "__main__":
    from langchain_openai import ChatOpenAI

    description = "I have a set of F1 driver resumes. I need to know what information is tracked (like stats and teams), what specific details are inside those categories (like wins or years), and how the drivers, teams, and awards are linked together."
    llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        base_url="https://api.groq.com/openai/v1",
        reasoning_effort="medium",
    )
    ontology_dict = {
        "entity_labels": ["Driver", "Team", "Award", "Statistic"],
        "relationship_rules": [
            {
                "source_label": "Driver",
                "relationship": "works_for",
                "target_label": "Team",
            },
            {
                "source_label": "Driver",
                "relationship": "has_statistic",
                "target_label": "Statistic",
            },
            {
                "source_label": "Statistic",
                "relationship": "belongs_to",
                "target_label": "Driver",
            },
            {
                "source_label": "Driver",
                "relationship": "awarded",
                "target_label": "Award",
            },
            {
                "source_label": "Award",
                "relationship": "associated_with",
                "target_label": "Team",
            },
        ],
    }
    ontology = Ontology(**ontology_dict)
    ex = GraphExtractor(
        description,
        [
            Document(
                "---\n\n### **Career Highlights**\n\n* **Formula 1 World Champion:** 1985, 1986, 1989, 1993\n* **First Driver to Reach 50 Grand Prix Wins**\n* **Multiple championship runner-up finishes** in ultra-competitive eras\n* Central figure in Formula 1's most iconic rivalry (late 1980s-early 1990s)\n* Championship success with **multiple constructors**\n\n---\n\n### **Formula 1 Career**\n\n**Renault (1980-1983)**"
            )
        ],
        llm,
        ontology,
    )
    ex.extract()
    print(ex.entities)
    print(ex.triplets)
