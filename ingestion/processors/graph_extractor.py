from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.prompts import PromptTemplate

load_dotenv()


class RelationshipRule(BaseModel):
    source_label: str
    relationship: str
    target_label: str


class Ontology(BaseModel):
    entity_labels: list[str]
    relationship_rules: list[RelationshipRule]


class Entity(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(..., description="Unique uuid identifier for the Entity")
    entity_label: str = Field(..., description="Label of the entity")
    properties: dict = Field(..., description="Properties belonging to the entity")


class Triplet(BaseModel):
    source_id: str = Field(description="Unique uuid of the source entity")
    relationship: str = Field(description="Relationship label between the entities")
    target_id: str = Field(description="Unique uuid of the target entity")


class EntityRelationships(BaseModel):
    entities: list[Entity] = Field(..., description="All identified entities")
    triplets: list[Triplet] = Field(
        ..., description="Tripltets of the identified entities"
    )


class GraphExtractor:
    def __init__(
        self,
        description: str,
        documents: list[Document],
        llm: BaseChatModel,
        ontology: Ontology = None,
    ):
        self.description = description
        self.documents = documents
        self.llm = llm
        self.ontology = ontology
        self._ontology_parser = PydanticOutputParser(pydantic_object=Ontology)
        self._triplet_parser = PydanticOutputParser(pydantic_object=EntityRelationships)

        self.entities: list[Entity] = []
        self.triplets: list[Triplet] = []

    def extract(self):
        if not self.ontology:
            self.ontology = self._extract_ontology()

        entity_storage: dict[str, Entity] = {}
        for document in self.documents:
            current_context_entities = list(entity_storage.values())
            extraction = self._apply_ontology_to_doc(document, current_context_entities)

            for ent in extraction.entities:
                if ent.id in entity_storage:
                    entity_storage[ent.id].properties.update(ent.properties)
                else:
                    entity_storage[ent.id] = ent
            self.triplets.extend(extraction.triplets)

        self.entities = list(entity_storage.values())

    def _extract_ontology(self) -> Ontology:
        ONTOLOGY_SYSTEM_PROMPT = f"""
        ### Role
        You are an Ontology Engineer. Your task is to analyze user descriptions and extract the structural schema for a knowledge graph in JSON format using the Subject-Predicate-Object (SPO) pattern.
        
        ### Objective
        Identify:
        1. **Entity Labels**: The categories of objects. (entity_labels).
        2. **Relationship Rules**: The valid connections between Entity Labels.

        ### Constraints
        - **Abstract Data Only**: Do not include specific instances or values (e.g., use "Product", not "iPhone 15").
        - **Directionality**: Every relationship must clearly define a 'source_label' and a 'target_label'.
        - **SPO Pattern**: Every relationship must be defined as a triple consisting of a Subject (source_label), a Predicate (relationship type), and an Object (target). The `source_label` and `target_label` must only be one of the entity_labels
        - **JSON Formatting**: Output must be a single, valid JSON object. No conversational filler.

        ### Strict JSON Output Format
        {self._ontology_parser.get_format_instructions()}
        """
        res = self.llm.invoke(
            [SystemMessage(ONTOLOGY_SYSTEM_PROMPT), HumanMessage(self.description)]
        )
        parsed: Ontology = self._ontology_parser.invoke(res.content)
        return parsed

    def _apply_ontology_to_doc(
        self, document: Document, existing_entities: list[Entity]
    ) -> EntityRelationships:
        EXTRACTION_SYSTEM_PROMPT = PromptTemplate.from_template(
            """
            ### Role
            You are an expert Knowledge Graph Engineer. Your task is to perform Named Entity Recognition (NER) and Relationship Extraction from user input based on a strictly defined ontology.

            ### Constraints
            - Ontology: Only extract the entities and relationships allowed in the given `entity_labels` and `relationship_rules`
                - `entity_labels` : {{entity_labels}}
                - `relationship_rules`: {{relationship_rules}}

            ### Existing Entities
            Below is a list of entities already identified in previous documents. 
            If the current text refers to these entities (even by pronoun or partial name), 
            REUSE their IDs instead of creating new ones.
            {{existing_entities}}

            ### Extraction Rules
            1. Strict Adherence: Extract ONLY the entity types listed in entity_labels. If an entity does not fit a label, ignore it. Assign a unique uuid to the extracted Entities.
            2. Relationship Validation: Only extract triples (Source - Relationship -> Target) that are explicitly permitted by the relationship_rules.
            3. Property Extraction: For each entity, capture relevant attributes from the text (e.g., "age", "location", "year") and place them inside the properties dictionary. If no properties are found, return an empty dictionary {}.
            4. Resolution: If the text refers to an entity by a pronoun (e.g., "he", "him") resolve it to an entity from the 'Existing Entities' list or a new entity you've identified in this document. If an existing entity exists then extend the properties
            5. Triplet construction: Only make the triplets from the identified list of entities, only use the ids of entities which have already been identified.
            6. Output Format: Output must be a single, valid JSON object. No conversational filler.

            ### Strict JSON Output Format
            {{output_format}}
            """,
            template_format="jinja2",
            partial_variables={
                "output_format": self._triplet_parser.get_format_instructions()
            },
        )
        sytem_prompt = EXTRACTION_SYSTEM_PROMPT.invoke(
            {
                "entity_labels": self.ontology.entity_labels,
                "relationship_rules": self.ontology.relationship_rules,
                "existing_entities": existing_entities,
            }
        ).to_string()
        res = self.llm.invoke(
            [SystemMessage(sytem_prompt), HumanMessage(document.page_content)]
        )
        parsed: EntityRelationships = self._triplet_parser.invoke(res)
        return parsed


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
