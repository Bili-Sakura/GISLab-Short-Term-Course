from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.globals import set_verbose, get_verbose
from langchain_glm import ChatZhipuAI
import os
from dotenv import load_dotenv
import json

load_dotenv()
# Get the API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
zhipuai_api_key = os.getenv("ZHIPUAI_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
set_verbose(value=True)  # Enable verbose mode


def generate_items():
    # Initialize the model
    # model = ChatOpenAI(model="glm-4-turbo", api_key="")
    # model = ChatOpenAI(model="gpt-3.5-turbo")
    model = ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_api_key,
        openai_api_base="https://api.deepseek.com",
        # max_tokens=1024,
    )
    # Create a prompt template
    system_template = "Generate 10 cities in JSON format with fields name and country:"
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "")]
    )

    # Initialize the parser
    parser = JsonOutputParser()

    # Chain the components together
    chain = prompt_template | model | parser
    response = chain.invoke({"text": ""})

    # Process the response
    items_json = json.dumps(response, indent=4)

    # Save the generated items to a JSON file
    with open("data/items.json", "w") as f:
        f.write(items_json)

    return items_json
