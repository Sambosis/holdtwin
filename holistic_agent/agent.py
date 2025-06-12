import logging
import time
from typing import Any, Dict, Optional, Tuple

from holistic_agent import config
from holistic_agent.action import ActionExecutor
from holistic_agent.memory import MemoryModule
from holistic_agent.model import DesktopFoundationModel, MultimodalTokenizer
from holistic_agent.perception import PerceptionModule
from holistic_agent.reasoning import ReasoningModule

# Set up a logger for the agent module
logger = logging.getLogger(__name__)


class HolisticAgent:
    """
    The central orchestrator of the Holistic Digital Twin Agent.

    This class manages the main "observe-think-act-memorize" loop. It coordinates
    the various modules (Perception, Reasoning, Action, Memory) to perceive
    the environment, decide on the next best action, execute it, and record
    the outcome.

    Attributes:
        config: The application configuration object.
        perception_module (PerceptionModule): The module for screen and text capture.
        reasoning_module (ReasoningModule): The module for planning and action generation.
        action_executor (ActionExecutor): The module for executing generated code.
        memory_module (MemoryModule): The module for long-term memory via Neo4j.
        model (DesktopFoundationModel): The core multimodal transformer model.
        tokenizer (MultimodalTokenizer): The tokenizer for processing vision and text data.
        agent_state (Dict[str, Any]): A dictionary to maintain the agent's current state,
                                      such as the high-level goal, task history, and status.
        is_running (bool): A flag to control the execution of the main loop.
    """

    def __init__(self) -> None:
        """
        Initializes the HolisticAgent and its constituent modules.

        This constructor sets up all necessary components by loading the configuration,
        instantiating the perception, memory, and action modules, and loading the
        core AI model and tokenizer to initialize the reasoning module.
        """
        self.config = config
        logger.info("Initializing HolisticAgent...")

        # Initialize core AI components
        self.model: DesktopFoundationModel = self._load_model()
        self.tokenizer: MultimodalTokenizer = self._load_tokenizer()

        # Initialize functional modules
        self.perception_module: PerceptionModule = PerceptionModule()
        self.memory_module: MemoryModule = MemoryModule()
        self.action_executor: ActionExecutor = ActionExecutor()
        self.reasoning_module: ReasoningModule = ReasoningModule(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_module=self.memory_module
        )

        self.agent_state: Dict[str, Any] = self._initialize_state()
        self.is_running: bool = False
        logger.info("HolisticAgent initialized successfully.")

    def _load_model(self) -> DesktopFoundationModel:
        """
        Loads the DesktopFoundationModel from the path specified in the config.

        Returns:
            DesktopFoundationModel: The instantiated core transformer model.
        """
        # In a real implementation, you would load a trained model from a file.
        # For now, we instantiate the placeholder model.
        # model = DesktopFoundationModel(config.ModelConfig)
        # model.load_state_dict(torch.load(self.config.MODEL_CHECKPOINT_PATH))
        # model.eval()
        # return model
        return DesktopFoundationModel(config.ModelConfig)

    def _load_tokenizer(self) -> MultimodalTokenizer:
        """
        Loads the MultimodalTokenizer from the path specified in the config.

        Returns:
            MultimodalTokenizer: The instantiated multimodal tokenizer.
        """
        # In a real implementation, you would load a tokenizer from a file.
        # For now, we instantiate the placeholder tokenizer.
        # return MultimodalTokenizer(vocab_path=self.config.TOKENIZER_PATH)
        return MultimodalTokenizer(vocab_path='')

    def _initialize_state(self) -> Dict[str, Any]:
        """
        Initializes the agent's state dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the initial agent state.
        """
        return {
            "current_goal": None,
            "task_history": [],
            "last_observation": None,
            "last_action_result": None,
            "status": "idle"
        }

    def run(self, user_goal: str) -> None:
        """
        Starts and manages the main observe-think-act-memorize loop.

        This method take a high-level user goal and continuously executes
        the agent's operational cycle until the goal is achieved, an error
        occurs, or the process is stopped manually.

        Args:
            user_goal (str): The high-level task for the agent to complete.
        """
        logger.info(f"Starting agent loop for goal: '{user_goal}'")
        self.agent_state["current_goal"] = user_goal
        self.is_running = True
        self.agent_state["status"] = "running"

        try:
            while self.is_running:
                # 1. OBSERVE
                pre_action_observation = self.agent_state.get("last_observation") or self._observe()

                # 2. THINK
                action_code, thought, is_task_complete = self._think(
                    pre_action_observation, None
                )

                if is_task_complete:
                    logger.info("Reasoning module determined the task is complete.")
                    self.stop()
                    break

                # 3. ACT
                execution_result = self._act(action_code)
                self.agent_state['last_action_result'] = execution_result

                # Capture state *after* the action for memory
                post_action_observation = self._observe()
                self.agent_state['last_observation'] = post_action_observation

                # 4. MEMORIZE
                self._memorize(
                    pre_action_observation,
                    thought,
                    action_code,
                    execution_result,
                    post_action_observation
                )

                time.sleep(1) # a delay to see the actions

        except Exception as e:
            logger.error(f"An unhandled exception occurred in the agent loop: {e}", exc_info=True)
            self.agent_state["status"] = "error"
        finally:
            if self.is_running:
                self.stop()
            logger.info("Agent loop has finished.")


    def _observe(self) -> Dict[str, Any]:
        """
        Performs the 'Observe' step of the cycle.

        Captures the current screen and extracts text using the PerceptionModule.

        Returns:
            Dict[str, Any]: A dictionary containing the observation data,
                            e.g., {'screenshot_path': 'path/to/img', 'ocr_text': '...'}.
        """
        logger.debug("Observing current screen state...")
        return self.perception_module.get_observation()

    def _think(self, observation: Dict[str, Any], context: Optional[str]) -> Tuple[str, str, bool]:
        """
        Performs the 'Think' step of the cycle.

        Uses the ReasoningModule to analyze the current state, goal, and memory
        context to decide on the next action using a ReAct-style process.

        Args:
            observation (Dict[str, Any]): The current observation from the _observe step.
            context (Optional[str]): Relevant historical context retrieved from memory.

        Returns:
            Tuple[str, str, bool]: A tuple containing:
                - The executable python code snippet for the next action.
                - The thought process or reasoning behind the action.
                - A boolean flag indicating if the task is considered complete.
        """
        logger.debug("Thinking about the next action...")
        action_code, thought = self.reasoning_module.generate_next_step(
            user_goal=self.agent_state["current_goal"],
            current_observation=observation
        )
        
        # For this placeholder version, we assume the task is never "complete" from thought alone.
        is_complete = False 
        
        self.agent_state["task_history"].append({"thought": thought, "action": action_code})
        return action_code, thought, is_complete

    def _act(self, action_code: str) -> Dict[str, Any]:
        """
        Performs the 'Act' step of the cycle.

        Executes the provided Python code snippet using the ActionExecutor.

        Args:
            action_code (str): The Python code to execute.

        Returns:
            Dict[str, Any]: A dictionary with the execution results, e.g.,
                            {'stdout': '...', 'stderr': '...', 'success': True}.
        """
        logger.info(f"Executing action code: \n---\n{action_code}\n---")
        success, result_msg = self.action_executor.execute_action(action_code)
        logger.debug(f"Action result: Success={success}, Msg={result_msg}")
        return {"success": success, "output": result_msg}

    def _memorize(
        self,
        pre_action_observation: Dict[str, Any],
        thought: str,
        action: str,
        result: Dict[str, Any],
        post_action_observation: Dict[str, Any]
    ) -> None:
        """
        Performs the 'Memorize' step of the cycle.

        Stores the completed interaction (state, thought, action, result, new_state)
        in the long-term memory module.

        Args:
            pre_action_observation (Dict[str, Any]): The state before the action.
            thought (str): The reasoning leading to the action.
            action (str): The action code that was executed.
            result (Dict[str, Any]): The result of the action execution.
            post_action_observation (Dict[str, Any]): The state after the action.
        """
        logger.debug("Memorizing the interaction...")
        self.memory_module.save_interaction(
            task_description=self.agent_state["current_goal"],
            previous_state_summary=pre_action_observation.get('ocr_text', ''),
            action_code=action,
            new_state_summary=post_action_observation.get('ocr_text', ''),
            metadata=result
        )

    def stop(self) -> None:
        """
        Stops the agent's main loop and performs cleanup, such as closing
        database connections.
        """
        if self.is_running:
            logger.info("Stopping the agent...")
            self.is_running = False
            self.agent_state["status"] = "stopped"
            # Ensure resources like database connections are released.
            self.memory_module.close()
            logger.info("Agent has been stopped.")
