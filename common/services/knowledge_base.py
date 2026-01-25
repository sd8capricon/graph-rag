from common.graph.client import GraphClient
from common.schema.knowledge_base import KnowledgeBase


class KnowledgeBaseService:

    def __init__(self, client: GraphClient):
        self.client = client

    def create_knowledge_base(self, knowledge_base: KnowledgeBase):
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

    def upsert_knowledge_base(self, knowledge_base: KnowledgeBase):
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
