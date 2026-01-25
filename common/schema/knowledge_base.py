from pydantic import BaseModel


class RelationshipRule(BaseModel):
    source_label: str
    relationship: str
    target_label: str


class Ontology(BaseModel):
    entity_labels: list[str]
    relationship_rules: list[RelationshipRule]


class KnowledgeBase(BaseModel):
    id: str
    name: str
    description: str | None = None
    knowledge_extraction_prompt: str | None = None
    ontology: Ontology | None = None
