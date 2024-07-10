# LangChain Project

This project uses LangChain with the OpenAI API to generate a batch of items in JSON format.

## Installation Guide

### Python Virtual Environment

1. **Create a Virtual Environment:**

    ```bash
    python -m venv .venv
    ```

2. **Activate the Virtual Environment:**
    - On Windows:

        ```bash
        .\.venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source .venv/bin/activate
        ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Conda Environment

1. **Create a Conda Environment:**

    ```bash
    conda create --name gpt python=3.10
    ```

2. **Activate the Conda Environment:**

    ```bash
    conda activate gpt
    ```

3. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Navigate to the Project Directory:**

    ```bash
    cd RSGPT
    ```

2. **Run the Main Script:**

    ```bash
    run.bat
    python main.py
    ```

This will generate a batch of items using LangChain with the OpenAI API and save them in `data/items.json`.

Create `.env` File

```
OPENAI_API_KEY="sk-xxx"
```
