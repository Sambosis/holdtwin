# Holistic Digital Twin Agent (holdtwin)

A generalist AI agent designed for desktop GUI automation. It perceives the screen, reasons about goals, executes actions, and learns from interactions using an observe-think-act-memorize loop.

## What it Does

The Holistic Digital Twin Agent automates desktop tasks by observing the graphical user interface (GUI), reasoning about the user's goal, executing actions (mouse clicks, keyboard inputs), and storing knowledge from these interactions. It operates on an **observe-think-act-memorize** cycle:

*   **Perception Module**: Captures screenshots and extracts text from the screen using Optical Character Recognition (OCR) to understand the current state of the GUI.
*   **Reasoning Module**: Utilizes a sophisticated AI model (Desktop Foundation Model) and a Multimodal Tokenizer. Given a user goal and the current observation, it determines the next best action or sequence of actions to perform. This often involves a ReAct-style (Reason+Act) prompting process.
*   **Action Module**: Executes the planned actions, typically as Python code snippets that interact with the GUI using libraries like PyAutoGUI.
*   **Memory Module**: Stores records of interactions (state, thought, action, result, new_state) in a Neo4j graph database. This allows the agent to learn from past experiences and improve its decision-making over time.

Core technologies include screen capture, OCR (Tesseract), GUI automation (PyAutoGUI), AI/machine learning models (custom Desktop Foundation Model, potentially augmented by LLMs like GPT), and a Neo4j graph database for long-term memory.

## Installation

1.  **Prerequisites:**
    *   Python >= 3.13

2.  **Clone the Repository:**
    ```bash
    git clone <repository_url>  # Replace <repository_url> with the actual URL
    cd holdtwin # Or your project's root directory name
    ```

3.  **Install Dependencies:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```
    Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Setup

Before running the agent, some setup is required:

1.  **Environment Variables:**
    Create a `.env` file in the root of the project directory. This file will store sensitive credentials and configuration. Add the following variables:

    ```env
    NEO4J_URI="neo4j://localhost:7687"
    NEO4J_USER="neo4j"
    NEO4J_PASSWORD="your_neo4j_password"

    # Optional: Path to Tesseract executable (especially for Windows)
    # TESSERACT_CMD_PATH="C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Optional: OpenAI API Key if using OpenAI models for advanced reasoning
    # OPENAI_API_KEY="your_openai_api_key"
    ```
    Replace `"your_neo4j_password"` and other placeholder values with your actual settings.

2.  **Neo4j Database:**
    Ensure you have a Neo4j database instance running and accessible with the URI, username, and password specified in your `.env` file. The agent uses Neo4j to store its long-term memory.

3.  **AI Models:**
    The agent requires a pre-trained Desktop Foundation Model and a Multimodal Tokenizer to function.
    *   **Model File:** `desktop_foundation_model.pt`
    *   **Tokenizer File:** `multimodal_tokenizer.json`

    These files must be placed in a `models/` directory in the project root (i.e., `<project_root>/models/desktop_foundation_model.pt`).

    **[IMPORTANT] How to Obtain the Models:**
    *   **[Placeholder: User needs to provide this information. e.g., Download from a specific URL, instructions to train, or a note if they are not publicly available.]**

4.  **Tesseract OCR:**
    The agent uses Tesseract OCR to read text from the screen. You need to install Tesseract on your system.
    *   Follow the official installation instructions for your operating system: [https://tesseract-ocr.github.io/tessdoc/Installation.html](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    *   Ensure that the Tesseract executable is in your system's PATH, or provide the full path via the `TESSERACT_CMD_PATH` environment variable in your `.env` file if needed (especially on Windows).
    *   You may also need to install language data for Tesseract (e.g., for English, `eng`).

## How to Use

Once the installation and setup are complete, you can run the agent from the project's root directory using the command line.

Provide the agent with a high-level goal as a string argument:

```bash
python holistic_agent/main.py "Your high-level goal for the agent"
```

**Example:**

To ask the agent to open Notepad and type "Hello, World!", you would run:

```bash
python holistic_agent/main.py "Open notepad and type Hello, World!"
```

The agent will then start its observe-think-act-memorize loop to attempt to achieve this goal.

**Logging:**
Log files are generated in the `<project_root>/logs/` directory. Check these logs for detailed information about the agent's operations and for troubleshooting.

## Project Structure

A brief overview of the key directories in this project:

```
holdtwin/
├── .env                # Local environment variables (you need to create this)
├── .gitignore          # Specifies intentionally untracked files that Git should ignore
├── holistic_agent/     # Core agent logic
│   ├── __init__.py
│   ├── action.py       # Action execution module
│   ├── agent.py        # Main agent class and observe-think-act loop
│   ├── config.py       # Configuration settings and environment variable loading
│   ├── main.py         # Command-line entry point
│   ├── memory.py       # Memory module (Neo4j interaction)
│   ├── model.py        # AI model definitions (Desktop Foundation Model)
│   ├── perception.py   # Screen perception and OCR module
│   └── reasoning.py    # Reasoning and planning module
├── logs/               # Directory for log files (created automatically)
├── models/             # Directory for AI model files (e.g., .pt, .json)
│                       # (You need to place model files here)
├── pyproject.toml      # Project metadata and dependencies for build systems
├── README.md           # This file
├── requirements.txt    # Python package dependencies
└── uv.lock             # Lock file for uv package manager (if used)
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, please feel free to open an issue or submit a pull request on the project's repository.

## License

This project does not currently have a license. It is recommended to add a license file (e.g., MIT, Apache 2.0) to clarify how others can use and contribute to this software.
