import os
import json
import yaml
import argparse
from tqdm import tqdm
from transformers import pipeline
from src.logger_utils import setup_logging

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Summarize captions using different models and configurations."
)
parser.add_argument(
    "--config", type=str, required=True, help="Path to the configuration YAML file."
)
args = parser.parse_args()

# Load configurations from the specified YAML file
with open(args.config, "r") as config_file:
    config = yaml.safe_load(config_file)

# Set up logging
log_file = config["logging"]["log_file"]
logger = setup_logging(log_file)
logger.info(f"Using configuration: {args.config}")
logger.info(f"Model description: {config['model']['description']}")
logger.info(f"Data description: {config['data']['description']}")
logger.info(f"Logging description: {config['logging']['description']}")
logger.info(f"Summarization description: {config['summarization']['description']}")

logger.info("Loading the model...")
try:
    model_path = config["model"]["path"]
    pipe = pipeline("text2text-generation", model=model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Define the few-shot examples from the config
few_shot_examples = config["summarization"]["few_shot_examples"]


# Function to format input with few-shot examples
def format_with_few_shot_examples(text, examples):
    prompt = "Examples:\n"
    for example in examples:
        prompt += f"Input: {example['input']}\nOutput: {example['output']}\n"
    prompt += f"Input: {text}\nOutput:"
    return prompt


# Function to summarize text using the model
def summarize_text(text):
    formatted_input = format_with_few_shot_examples(text, few_shot_examples)
    try:
        summary = pipe(formatted_input)[0]["generated_text"]
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return ""


# File paths from the config
input_file = config["data"]["input_file"]
output_file = config["data"]["output_file"]


# Function to load processed images
def load_processed_images(output_file):
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                data = json.loads(line)
                processed_images.add(data["image"])
    return processed_images


processed_images_set = load_processed_images(output_file)

with open(input_file, "r") as reader:
    total_lines = sum(1 for _ in reader)  # Count total lines for tqdm
    reader.seek(0)  # Reset file pointer to the beginning
    with open(output_file, "a") as writer:
        for line in tqdm(reader, total=total_lines, desc="Summarizing Captions"):
            obj = json.loads(line.strip())
            image_filename = obj.get("image")
            # Skip if the image has already been processed
            if image_filename in processed_images_set:
                continue
            caption = obj.get("caption", "")
            caption_short = summarize_text(caption)
            # Add the summarized caption to the object
            obj["caption_short"] = caption_short
            logger.info(f"Image: {image_filename}, Short_Caption: {caption_short}")
            # Write the updated object to the output file
            writer.write(json.dumps(obj) + "\n")
            writer.flush()

logger.info(f"Processed and saved summarized captions to {output_file}")
