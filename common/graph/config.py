from common.utils.environment import require_env


def neo4j_config():
    return dict(
        username=require_env("NEO4J_USER"),
        password=require_env("NEO4J_PASSWORD"),
        url=require_env("NEO4J_URI"),
        embedding_dimension=768,
    )
