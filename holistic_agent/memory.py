import logging
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase, Session, Transaction

from holistic_agent import config

# Configure logging for the memory module
logger = logging.getLogger(__name__)


class MemoryModule:
    """
    Manages the connection to the Neo4j temporal knowledge graph.

    This class serves as a dedicated interface for all database operations,
    encapsulating the logic for storing agent experiences and retrieving
    historical context to inform decision-making. It handles the connection
    lifecycle and provides high-level methods for interaction with the graph.
    """

    def __init__(self) -> None:
        """
        Initializes the MemoryModule and establishes a connection to Neo4j.

        The constructor reads connection details (URI, user, password) from the
        global configuration file and instantiates the Neo4j driver. It also
        verifies the connection to ensure the database is reachable.
        """
        self._driver: Optional[Driver] = None
        try:
            self._driver = GraphDatabase.driver(
                config.MemoryConfig.URI,
                auth=(config.MemoryConfig.USER, config.MemoryConfig.PASSWORD)
            )
            self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            # In a real application, this might raise the exception or
            # enter a degraded state.
            pass

    def save_interaction(
        self,
        task_description: str,
        previous_state_summary: str,
        action_code: str,
        new_state_summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Saves a complete interaction cycle to the knowledge graph.

        This method creates nodes for the states and the action, and connects
        them in a temporal sequence. It links the interaction to a parent
        'Task' node, allowing for task-based context retrieval.

        The graph structure typically looks like:
        (Task)-[:HAS_STEP]->(PrevState)-[:EXECUTED]->(Action)-[:RESULTED_IN]->(NewState)

        Args:
            task_description (str): A description of the high-level goal the
                agent is currently pursuing.
            previous_state_summary (str): A textual summary of the environment
                state before the action was taken.
            action_code (str): The Python code snippet that was executed.
            new_state_summary (str): A textual summary of the environment
                state after the action was executed.
            metadata (Optional[Dict[str, Any]]): A dictionary containing any
                extra information to store, such as timestamps, success flags,
                or execution duration. Defaults to None.
        """
        params = {
            "task_desc": task_description,
            "prev_state": previous_state_summary,
            "action_code": action_code,
            "new_state": new_state_summary,
            "metadata": metadata or {}
        }
        query = (
            "MERGE (t:Task {description: $task_desc}) "
            "CREATE (ps:State {summary: $prev_state}) "
            "CREATE (a:Action {code: $action_code}) "
            "CREATE (ns:State {summary: $new_state}) "
            "MERGE (t)-[:HAS_STEP]->(ps) "
            "MERGE (ps)-[:EXECUTED]->(a) "
            "MERGE (a)-[:RESULTED_IN]->(ns)"
        )
        self._execute_query(query, params, write=True)

    def query_context_by_task(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieves historical interactions related to a specific task.

        This method queries the graph for past sequences of (State, Action)
        that belong to tasks with a similar description. This context can be
        used by the ReasoningModule to form better plans.

        Args:
            task_description (str): The description of the current task to find
                relevant history for.
            limit (int): The maximum number of historical interactions to return.
                Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
            represents a past interaction, containing keys like 'previous_state',
            'action', and 'new_state'. Returns an empty list if no relevant
            history is found.
        """
        query = (
            "MATCH (t:Task)-[:HAS_STEP]->(ps:State)-[:EXECUTED]->(a:Action)-[:RESULTED_IN]->(ns:State) "
            "WHERE t.description CONTAINS $task_desc "
            "RETURN ps.summary AS previous_state, a.code AS action, ns.summary AS new_state "
            "ORDER BY a.timestamp DESC LIMIT $limit"
        )
        params = {"task_desc": task_description, "limit": limit}
        return self._execute_query(query, params)

    def find_similar_states(
        self,
        state_summary: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Finds past states similar to the given state summary.

        This can be used to retrieve actions that were successful in similar
        situations in the past, aiding in action generation.

        Args:
            state_summary (str): The summary of the current state.
            limit (int): The maximum number of similar states to return.
                Defaults to 3.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a
            historically similar state and the action that followed it.
        """
        # This is a simplified example. A real implementation would use text similarity
        # algorithms or graph-based similarity metrics.
        query = (
            "MATCH (s:State) "
            "WHERE s.summary CONTAINS $summary_keyword "
            "OPTIONAL MATCH (s)-[:EXECUTED]->(a:Action) "
            "RETURN s.summary AS state, a.code AS next_action, id(s) as node_id "
            "LIMIT $limit"
        )
        # A basic keyword extraction for demonstration
        keyword = state_summary.split(' ')[0]

        params = {"summary_keyword": keyword, "limit": limit}
        return self._execute_query(query, params)

    def close(self) -> None:
        """
        Gracefully closes the connection to the Neo4j database.

        It's important to call this method upon application shutdown to release
        database resources properly.
        """
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def _execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        write: bool = False
    ) -> List[Dict[str, Any]]:
        """
        A private helper method to execute a Cypher query.

        This method abstracts the session and transaction management for both
        read and write operations, providing a single point for query execution.

        Args:
            query (str): The Cypher query to be executed.
            params (Optional[Dict[str, Any]]): A dictionary of parameters to
                pass to the query. Defaults to None.
            write (bool): If True, the query is executed in a write transaction.
                Otherwise, it's executed in a read transaction. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of records returned by the query, where
            each record is a dictionary.
        """
        if not self._driver:
            logger.error("Driver not initialized. Cannot execute query.")
            return []

        params = params or {}
        with self._driver.session() as session:
            try:
                if write:
                    result = session.execute_write(self._run_transaction, query, params)
                else:
                    result = session.execute_read(self._run_transaction, query, params)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"Error executing Cypher query: {e}", exc_info=True)
                return []

    @staticmethod
    def _run_transaction(
        tx: Transaction,
        query: str,
        params: Dict[str, Any]
    ) -> Any:
        """
        Callback function to run a query within a transaction.

        Args:
            tx (Transaction): The Neo4j transaction object.
            query (str): The Cypher query string.
            params (Dict[str, Any]): The parameters for the query.

        Returns:
            Any: The result of the query execution.
        """
        result = tx.run(query, **params)
        return result
