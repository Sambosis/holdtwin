"""
reasoning.py

Implements the advanced reasoning capabilities for the Holistic Digital Twin Agent.

This module contains the ReasoningModule, which is responsible for implementing
the "Reason and Act" (ReAct) prompting strategy. It takes a high-level user
request and the current state of the system (observation), then interacts with
the core multimodal model to break the problem down into a sequence of
thought-action steps. The module orchestrates the generation of reasoning traces
and the extraction of a final, executable action for the current step in a task.
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

# Conditional import for type hinting to avoid circular dependencies
if TYPE_CHECKING:
    from holistic_agent.memory import MemoryModule
    from holistic_agent.model import DesktopFoundationModel, MultimodalTokenizer


class ReasoningModule:
    """
    Orchestrates the ReAct (Reason and Act) loop to drive agent behavior.

    This class forms the cognitive core of the agent. It takes high-level goals
    and uses the foundation model to generate a sequence of thoughts and actions
    to achieve those goals. It formats prompts, queries for relevant memories,
    interprets model outputs, and maintains the history of the current reasoning
    process.
    """

    def __init__(
        self,
        model: "DesktopFoundationModel",
        tokenizer: "MultimodalTokenizer",
        memory_module: "MemoryModule",
    ):
        """
        Initializes the ReasoningModule.

        Args:
            model (DesktopFoundationModel): The core multimodal transformer model used
                for generating thoughts and actions.
            tokenizer (MultimodalTokenizer): The tokenizer corresponding to the model,
                used for preparing inputs.
            memory_module (MemoryModule): The interface to the long-term memory
                (Neo4j graph) to retrieve contextual information.
        """
        self.model: "DesktopFoundationModel" = model
        self.tokenizer: "MultimodalTokenizer" = tokenizer
        self.memory_module: "MemoryModule" = memory_module
        self.reasoning_history: List[Dict[str, str]] = []
        self.logger = logging.getLogger(__name__)

    def generate_next_step(
        self,
        user_goal: str,
        current_observation: Dict[str, Any],
        max_retries: int = 3,
    ) -> Tuple[str, str]:
        """
        Generates a thought and a corresponding executable action for the next step.

        This is the main entry point for the reasoning process. It orchestrates
        the creation of a ReAct prompt, retrieves context from memory, calls the
        model, and parses the output to get the next thought and action. It also
        updates the internal reasoning history.

        Args:
            user_goal (str): The high-level objective provided by the user.
            current_observation (Dict[str, Any]): A dictionary containing the current
                state of the environment, e.g., {'image': PIL.Image, 'ocr_text': str}.
            max_retries (int): The maximum number of attempts to generate a valid
                thought-action pair from the model.

        Returns:
            Tuple[str, str]: A tuple containing the generated thought (the reasoning
            behind the action) and the executable Python code snippet (the action).
            Returns (None, None) if a valid action cannot be generated.
        """
        # TODO: Implement retry logic and the main loop.
        historical_context = self._query_memory_for_context(
            user_goal, current_observation.get("ocr_text", "")
        )

        prompt_input = self._build_prompt(
            user_goal, current_observation, historical_context
        )

        # Tokenize the prompt and image for the model
        # model_input = self.tokenizer.encode(prompt_input, current_observation['image'])

        # Generate output from the model
        # raw_model_output = self.model.generate(model_input)
        raw_model_output = "Thought: I should click on the notepad icon.\nAction: pyautogui.click(x=500, y=500)" # Placeholder for model output
        thought, action = self._parse_model_output(raw_model_output)

        if thought and action:
            self.reasoning_history.append({"thought": thought, "action": action})
            self.logger.info(f"Generated thought: {thought}")
            self.logger.info(f"Generated action: {action}")
        else:
            self.logger.warning("Failed to parse a valid thought-action pair.")
            return None, None

        return thought, action

    def _build_prompt(
        self,
        user_goal: str,
        current_observation: Dict[str, Any],
        historical_context: str,
    ) -> str:
        """
        Constructs the full ReAct prompt to be sent to the foundation model.

        This method assembles all necessary pieces of information—the goal,
        historical context from memory, the current reasoning chain, and the
        current observation—into a single, structured prompt that guides the
        model to produce a "Thought" and an "Action".

        Args:
            user_goal (str): The high-level user objective.
            current_observation (Dict[str, Any]): The current state, including OCR text.
            historical_context (str): A summary of relevant past experiences from memory.

        Returns:
            str: A formatted prompt string ready for the model's tokenizer.
        """
        prompt = f"""You are a helpful assistant that controls a computer.
        You are given a high-level goal and the current screen content (as OCR text).
        You need to reason about the next action to take.

        High-level goal: {user_goal}

        Relevant past experiences:
        {historical_context}

        Current reasoning trace:
        """
        for step in self.reasoning_history:
            prompt += f"Thought: {step['thought']}\nAction: {step['action']}\n"
        
        prompt += f"\nCurrent screen content:\n{current_observation.get('ocr_text', '')}\n"
        prompt += "Thought:"
        return prompt

    def _parse_model_output(self, model_output: str) -> Tuple[str, str] | Tuple[None, None]:
        """
        Parses the raw text output from the model to extract thought and action.

        The model is expected to generate text in a specific format, e.g.:
        "Thought: I need to click the 'File' menu to find the save option.
         Action: pyautogui.click(x=50, y=30)"
        This method uses string manipulation or regex to extract these two components.

        Args:
            model_output (str): The raw string generated by the language model.

        Returns:
            Tuple[str, str] | Tuple[None, None]: A tuple containing the extracted
            thought and action code. Returns (None, None) if parsing fails.
        """
        try:
            thought_marker = "Thought:"
            action_marker = "Action:"
            
            thought_start = model_output.find(thought_marker)
            action_start = model_output.find(action_marker)

            if thought_start == -1 or action_start == -1:
                return None, None

            thought = model_output[thought_start + len(thought_marker):action_start].strip()
            action = model_output[action_start + len(action_marker):].strip()
            
            return thought, action
        except Exception as e:
            self.logger.error(f"Error parsing model output: {e}")
            return None, None

    def _query_memory_for_context(
        self, user_goal: str, current_observation_text: str
    ) -> str:
        """
        Queries the memory module for context relevant to the current task.

        Uses the user goal and text from the current observation to find similar
        past interactions stored in the Neo4j knowledge graph. The retrieved
        information is then formatted into a concise string to be included in the
        prompt.

        Args:
            user_goal (str): The overall task goal.
            current_observation_text (str): The text extracted via OCR from the
                current screen view.

        Returns:
            str: A string summarizing relevant memories, or an empty string if
                 none are found.
        """
        try:
            # Query for tasks with similar descriptions
            task_context = self.memory_module.query_context_by_task(user_goal, limit=2)
            
            # Query for states with similar content
            state_context = self.memory_module.find_similar_states(current_observation_text, limit=2)

            # Format the context for the prompt
            context_str = ""
            if task_context:
                context_str += "\nPast actions for similar tasks:\n"
                for item in task_context:
                    context_str += f"- In a similar state, the action taken was: `{item['action']}`\n"
            
            if state_context:
                context_str += "\nActions taken from similar states:\n"
                for item in state_context:
                    if item.get('next_action'):
                        context_str += f"- From a state like this, the action was: `{item['next_action']}`\n"

            return context_str if context_str else "No relevant memories found."
        except Exception as e:
            self.logger.error(f"Error querying memory for context: {e}")
            return "Error retrieving memories."

    def reset_history(self) -> None:
        """
        Resets the internal reasoning history.

        This should be called at the beginning of a new, independent user task to
        ensure the reasoning chain starts fresh.
        """
        self.reasoning_history = []
        self.logger.info("Reasoning history has been reset.")
