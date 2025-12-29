from pydantic import BaseModel, ConfigDict, Field


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
    doc_ids: set[str] = Field(default_factory=set, exclude=True)


class Triplet(BaseModel):
    source_id: str = Field(description="Unique uuid of the source entity")
    relationship: str = Field(description="Relationship label between the entities")
    target_id: str = Field(description="Unique uuid of the target entity")


class EntityRelationships(BaseModel):
    entities: list[Entity] = Field(..., description="All identified entities")
    triplets: list[Triplet] = Field(
        ..., description="Tripltets of the identified entities"
    )
