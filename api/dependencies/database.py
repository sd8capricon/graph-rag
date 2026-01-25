from common.graph.client import GraphClient

_neo4j_connection: GraphClient | None = None


def set_neo4j_connection(client: GraphClient):
    global _neo4j_connection
    _neo4j_connection = client


def close_neo4j_connection():
    global _neo4j_connection
    if _neo4j_connection:
        _neo4j_connection.close()
        _neo4j_connection = None


def get_neo4j_connection() -> GraphClient:
    if not _neo4j_connection:
        raise ValueError("Neo4j connection not initialized")
    return _neo4j_connection
