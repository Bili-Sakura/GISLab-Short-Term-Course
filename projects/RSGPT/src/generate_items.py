from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.globals import set_verbose, get_verbose
import os
from dotenv import load_dotenv
import json

load_dotenv()
set_verbose(value=True)  # Enable verbose mode


def generate_items():
    # Initialize the model
    model = ChatOpenAI(model="gpt-3.5-turbo")

    # Create a prompt template
    system_template = "Generate a list of items:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "")]
    )

    # Initialize the parser
    parser = StrOutputParser()

    # Chain the components together
    chain = prompt_template | model | parser
    response = chain.invoke({"text": ""})

    # Process the response
    items = response.split("\n")  # Assuming each item is on a new line
    items_json = json.dumps(items, indent=4)

    # Save the generated items to a JSON file
    with open("data/items.json", "w") as f:
        f.write(items_json)

    return items_json
