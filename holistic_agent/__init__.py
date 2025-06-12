# -*- coding: utf-8 -*-
"""
Holistic Digital Twin Agent Package.

This package provides the core modules for building and running a generalist AI agent
capable of operating a standard desktop graphical user interface. By unifying
perception, reasoning, and action into a single framework, it aims to create
a "digital twin" that can understand and execute complex, multi-step tasks.

The main entry point for using the agent's capabilities is the `HolisticAgent`
class, which orchestrates the various components defined in the submodules.

Modules:
    - agent: The central coordinator of the observe-think-act loop.
    - action: The executor for generated code, controlling mouse and keyboard.
    - config: Centralized configuration settings for the entire project.
    - memory: Interface to the long-term knowledge graph (Neo4j).
    - model: The core multimodal transformer model (PyTorch).
    - perception: Handles screen capture and OCR.
    - reasoning: Implements advanced prompting and planning strategies.
"""

import logging

# By defining a null handler, we prevent library-level log messages from propagating
# to the root logger's handlers unless the application using the library
# configures its own logging. This is the recommended practice for libraries.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Define the package version. Follows semantic versioning.
# See: https://semver.org/
__version__ = "0.1.0"
