from typing import Any, Dict, List, Optional
from neo4j import GraphDatabase, Driver, Session


class GraphClient:
    """
    Thin graph database client.
    Responsibilities:
    - connection management
    - executing Cypher queries
    - returning raw results

    NO business logic.
    NO RAG logic.
    NO ingestion rules.
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: Optional[str] = None,
        max_connection_lifetime: int = 3600,
        max_connection_pool_size: int = 50,
    ):
        self._driver: Driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            max_connection_lifetime=max_connection_lifetime,
            max_connection_pool_size=max_connection_pool_size,
        )
        self._database = database

    def get_driver(self) -> Driver:
        return self._driver

    def get_session(self) -> Session:
        """
        Create and return a new session.
        Caller is responsible for closing it.
        """
        return self._driver.session()

    def close_session(self, session: Session) -> None:
        """
        Safely close a session.
        """
        if session:
            session.close()

    def close(self) -> None:
        self._driver.close()

    def run(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results as a list of dicts.
        """
        parameters = parameters or {}

        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

    def run_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Execute a write-only query.
        """
        parameters = parameters or {}

        with self._driver.session(database=self._database) as session:
            session.run(query, parameters)

    def health_check(self) -> bool:
        """
        Simple connectivity check.
        """
        try:
            self.run("RETURN 1 AS ok")
            return True
        except Exception:
            return False
