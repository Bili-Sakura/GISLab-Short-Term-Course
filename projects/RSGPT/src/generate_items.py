from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.globals import set_verbose, get_verbose
import os
from dotenv import load_dotenv
import json

load_dotenv()
# Get the API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
set_verbose(value=True)  # Enable verbose mode


def generate_items():
    # Initialize the model
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com",
        # max_tokens=1024,
    )
    # Create a prompt template
    system_template = "Generate 10 cities in JSON format with fields name, country, latitude, and longitude:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "")]
    )

    # Initialize the parser
    parser = JsonOutputParser()

    # Chain the components together
    chain = prompt_template | model | parser
    response = chain.invoke({"text": ""})

    # Process the response
    cities_json = json.dumps(response, indent=4)

    # Save the generated items to a JSON file
    with open("data/cities.json", "w") as f:
        f.write(cities_json)

    return cities_json


