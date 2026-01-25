import json

from common.graph.client import GraphClient
from common.schema.knowledge_base import KnowledgeBase, Ontology


class KnowledgeBaseService:

    def __init__(self, client: GraphClient):
        self.client = client

    def create(self, knowledge_base: KnowledgeBase):
        query = """//cypher
        CREATE (kb:KnowledgeBase {
          id: $id,
          name: $name,
          description: $description,
          knowledge_extraction_prompt: $knowledge_extraction_prompt,
          ontology: $ontology,
          createdAt: datetime()
        })
        RETURN kb
        """
        parameters = {
            "id": knowledge_base.id,
            "name": knowledge_base.name,
            "description": knowledge_base.description,
            "knowledge_extraction_prompt": knowledge_base.knowledge_extraction_prompt,
            "ontology": (
                knowledge_base.ontology.model_dump_json()
                if knowledge_base.ontology
                else None
            ),
        }

        self.client.run_write(query, parameters)

    def get_ontology_by_id(self, knowledge_base_id: str) -> Ontology | None:
        query = """//cypher
        MATCH (kb:KnowledgeBase {id: $id})
        RETURN kb.ontology as ontology
        """
        parameters = {
            "id": knowledge_base_id,
        }

        result = self.client.run(query, parameters)

        if not result:
            return None

        ontology_json = result[0].get("ontology")

        if not ontology_json:
            return None

        return Ontology.model_validate(json.loads(ontology_json))

    def get_by_id(self, id: str) -> KnowledgeBase | None:
        query = """//cypher
        MATCH (kb:KnowledgeBase {id: $id})
        RETURN kb
        """
        parameters = {
            "id": id,
        }

        result = self.client.run(query, parameters)

        if not result:
            return None

        kb_data = result[0]["kb"]
        ontology_json = kb_data.get("ontology")

        ontology = None
        if ontology_json:
            ontology = Ontology.model_validate(json.loads(ontology_json))

        return KnowledgeBase(
            id=kb_data["id"],
            name=kb_data["name"],
            description=kb_data.get("description"),
            knowledge_extraction_prompt=kb_data.get("knowledge_extraction_prompt"),
            ontology=ontology,
        )

    def upsert(self, knowledge_base: KnowledgeBase):
        query = """//cypher
        MERGE (kb:KnowledgeBase {id: $id})
        ON CREATE SET
          kb.name = $name,
          kb.description = $description,
          kb.knowledge_extraction_prompt = $knowledge_extraction_prompt,
          kb.ontology = $ontology,
          kb.createdAt = datetime()
        ON MATCH SET
          kb.name = $name,
          kb.description = $description,
          kb.knowledge_extraction_prompt = $knowledge_extraction_prompt,
          kb.ontology = $ontology,
          kb.updatedAt = datetime()
        RETURN kb
        """
        parameters = {
            "id": knowledge_base.id,
            "name": knowledge_base.name,
            "description": knowledge_base.description,
            "knowledge_extraction_prompt": knowledge_base.knowledge_extraction_prompt,
            "ontology": (
                knowledge_base.ontology.model_dump_json()
                if knowledge_base.ontology
                else None
            ),
        }

        self.client.run_write(query, parameters)
