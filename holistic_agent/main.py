# -*- coding: utf-8 -*-
"""
Main entry point for the Holistic Digital Twin Agent.

This script initializes all necessary components, loads the configuration,
parses command-line arguments, and starts the agent's main operational loop.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import sys
from typing import List

from holistic_agent import config
from holistic_agent.agent import HolisticAgent

# Configure a basic logger for the application entry point
# A more sophisticated configuration might be loaded from the config module.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    """
    Parses command-line arguments for the agent application.

    Args:
        argv (List[str]): The list of command-line arguments, typically sys.argv[1:].

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
                            It will have a 'goal' attribute with the user's task.
    """
    parser = argparse.ArgumentParser(
        description="Holistic Digital Twin Agent: A generalist AI agent for desktop GUI automation."
    )
    parser.add_argument(
        "goal",
        type=str,
        help="The high-level user goal for the agent to accomplish. This should be a clear, natural language instruction."
    )
    # TODO: Add other potential arguments like --config-file, --verbose, etc.
    return parser.parse_args(args=argv)


def main(cli_args: List[str]) -> int:
    print("Starting main function...")
    logger.info("Starting Holistic Digital Twin Agent...")

    try:
        args = parse_arguments(cli_args)
        user_goal = args.goal
        logger.info(f"Received user goal: '{user_goal}'")

        # Initialize the main agent class. The agent's constructor is expected
        # to set up all its required modules (Perception, Action, Memory, etc.)
        # using settings from the centralized config module.
        logger.info("Initializing agent components...")
        agent = HolisticAgent()

        # Start the agent's primary control loop. The `run` method will
        # contain the observe-think-act-memorize loop and will continue
        # until the task is complete, fails, or is interrupted.
        logger.info("Starting agent's main control loop...")
        agent.run(user_goal=user_goal)

        logger.info("Agent has completed its task successfully.")
        return 0

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Shutting down gracefully.")
        # TODO: Implement any necessary cleanup logic here.
        return 1

    except Exception as e:
        logger.critical(f"An unhandled exception occurred: {e}", exc_info=True)
        # exc_info=True logs the full stack trace for debugging.
        return 1


if __name__ == "__main__":
    # Pass command-line arguments (excluding the script name) to the main function
    # and use its return value as the exit code.
    sys.exit(main(sys.argv[1:]))
