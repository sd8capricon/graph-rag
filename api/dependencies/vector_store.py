from langchain_neo4j.vectorstores.neo4j_vector import Neo4jVector

from api.services.vectorstore import VectorStoreService


# Functions for dependency injection
def get_community_store() -> Neo4jVector:
    return VectorStoreService().get_community_store()


def get_lexical_store() -> Neo4jVector:
    return VectorStoreService().get_lexical_store()
