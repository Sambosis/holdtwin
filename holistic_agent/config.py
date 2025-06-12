# -*- coding: utf-8 -*-
"""
holistic_agent/config.py

Global configuration settings for the Holistic Digital Twin Agent.

This module centralizes all configuration variables for the project, making it
easy to manage and adjust parameters for different environments (development,
production) without modifying the core logic of the application.

It is recommended to use a `.env` file in the project root to store sensitive
information like API keys and database credentials. This module will automatically
load variables from that file.

Example `.env` file:
    NEO4J_URI="neo4j://localhost:7687"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="your_secure_password"
    OPENAI_API_KEY="your_optional_openai_key"
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from a .env file at the project root.
# This allows for secure management of secrets like API keys and passwords.
# The project root is assumed to be the parent directory of 'holistic_agent'.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")


# --- General Settings ---
PROJECT_NAME: str = "HolisticDigitalTwinAgent"
AGENT_VERSION: str = "0.1.0"


# --- Path Configuration ---
# Absolute paths are constructed from the project root to ensure consistency.
LOGS_DIR: Path = PROJECT_ROOT / "logs"
MODEL_DIR: Path = PROJECT_ROOT / "models"
MODEL_CHECKPOINT_PATH: Path = MODEL_DIR / "desktop_foundation_model.pt"
TOKENIZER_PATH: Path = MODEL_DIR / "multimodal_tokenizer.json"

# On some systems, especially Windows, pytesseract needs the explicit path.
# Set this in your .env file or modify the default here if needed.
TESSERACT_CMD_PATH: str | None = os.getenv("TESSERACT_CMD_PATH")


# --- Logging Configuration ---
# Provides standardized logging settings for the entire application.
LOG_LEVEL: int = logging.INFO
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILENAME: str = f"{PROJECT_NAME.lower()}.log"
LOG_FILE_PATH: Path = LOGS_DIR / LOG_FILENAME


# --- Perception Module Settings ---
# Configuration for screen capture and OCR processing.
class PerceptionConfig:
    """Namespace for perception-related configuration."""
    # The monitor number to capture. 0 is the primary display.
    CAPTURE_MONITOR: int = 1

    # Language for Tesseract OCR. 'eng' for English.
    # Use 'eng+fra' for multiple languages.
    OCR_LANGUAGE: str = "eng"

    # The size (in pixels) of the square patches the screen image is divided into.
    # This must match the patch size the Vision Transformer model was trained on.
    IMAGE_PATCH_SIZE: int = 16


# --- Model Settings ---
# Configuration for the core multimodal transformer model.
class ModelConfig:
    """Namespace for model architecture and training configuration."""
    # Maximum number of tokens (image patches + text tokens) in a sequence.
    MAX_SEQUENCE_LENGTH: int = 4096

    # The embedding dimension for the transformer model.
    EMBEDDING_DIM: int = 768

    # The number of layers in the transformer encoder.
    NUM_LAYERS: int = 12

    # The number of attention heads in the multi-head attention mechanism.
    NUM_HEADS: int = 12
    IMAGE_PATCH_SIZE: int = 16
    ffn_dim: int = 3072 # Feed-forward network dimension
    vocab_size: int = 50257



# --- Action Executor Settings ---
# Configuration for the safe execution of agent-generated code.
class ActionExecutorConfig:
    """Namespace for action execution configuration."""
    # Default timeout in seconds for running a generated Python script.
    # This prevents the agent from getting stuck on a failing or long-running action.
    EXECUTION_TIMEOUT: int = 30

    # Global pause in seconds after each pyautogui function call.
    # This helps make GUI automation more reliable.
    PYAUTOGUI_PAUSE: float = 0.1


# --- Memory Module Settings (Neo4j) ---
# Configuration for connecting to the Neo4j temporal knowledge graph.
# It is STRONGLY recommended to set these values in your .env file.
class MemoryConfig:
    """Namespace for Neo4j database connection settings."""
    # The connection URI for the Neo4j database.
    URI: str = os.getenv("NEO4J_URI", "neo4j://localhost:7687")

    # The username for authenticating with the Neo4j database.
    USER: str = os.getenv("NEO4J_USER", "neo4j")

    # The password for authenticating with the Neo4j database.
    PASSWORD: str = os.getenv("NEO4J_PASSWORD", "")


# --- Reasoning Module Settings ---
# Configuration for ReAct prompting and interaction with language models.
class ReasoningConfig:
    """Namespace for reasoning and prompting configuration."""
    # API key for an external LLM like GPT-4, if used for advanced reasoning.
    # Can be left empty if the internal model handles all reasoning.
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

    # The model identifier to use for reasoning (e.g., "gpt-4-turbo").
    REASONING_MODEL_ID: str = "gpt-4-turbo"

    # Maximum number of steps in a single ReAct reasoning loop to prevent infinite loops.
    MAX_REASONING_STEPS: int = 15
