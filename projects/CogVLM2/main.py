import os
import json
from tqdm import tqdm
from PIL import Image
from src.config import MODEL_PATH, DEVICE, TORCH_TYPE
from src.model_utils import initialize_model, load_model, get_device_map
from src.captioning_utils import generate_caption
from src.logger_utils import setup_logging

UNLIMITED_NUM = 99999999
DISASTER_TYPES = ["volcano", "hurricane", "tornado", "earthquake", "flooding", "tsunami", "fire"]

def extract_disaster_type(folder_name):
    for disaster in DISASTER_TYPES:
        if disaster in folder_name.lower():
            return disaster
    return "unknown"

def extract_phrase(file_name):
    if "post_disaster" in file_name:
        return "post"
    elif "pre_disaster" in file_name:
        return "pre"
    else:
        return "unknown"

def load_processed_images(output_file):
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                processed_images.add(data['image'])
    return processed_images

def process_images(image_dir, output_file, logger, max_images=10):
    # Initialize the tokenizer and model
    tokenizer, model = initialize_model(MODEL_PATH, TORCH_TYPE)

    # Infer device map and load the model across GPUs
    device_map = get_device_map(model, max_memory_per_gpu="20GiB", num_gpus=3)
    model = load_model(model, MODEL_PATH, device_map, TORCH_TYPE)
    model = model.eval()

    # Load already processed images
    processed_images_set = load_processed_images(output_file)

    processed_images = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        # Open the output file in append mode for streaming write
        with open(output_file, 'a') as f:
            # Traverse through the directory structure and process each image
            for disaster_dir in tqdm(os.listdir(image_dir), desc="Processing disasters"):
                disaster_path = os.path.join(image_dir, disaster_dir, 'images')
                if os.path.isdir(disaster_path):
                    disaster_type = extract_disaster_type(disaster_dir)

                    # Process each image
                    for image_file in tqdm(os.listdir(disaster_path), desc=f"Processing images in {disaster_dir}", leave=False):
                        if image_file.endswith(".png"):
                            # Skip if the image has already been processed
                            if image_file in processed_images_set:
                                continue

                            phrase = extract_phrase(image_file)
                            prompt_template = (
                                f"Describe this {phrase}-event satellite image in detail. "
                                f"The disaster type is {disaster_type}. Give a description in one paragraph. "
                                "The description should be in the style of a news article and should be informative and actionable."
                            )
                            image_path = os.path.join(disaster_path, image_file)
                            try:
                                image = Image.open(image_path).convert("RGB")
                                caption = generate_caption(model, tokenizer, image, prompt_template, DEVICE, TORCH_TYPE)
                                result = {
                                    "disaster": disaster_dir,
                                    "image": image_file,
                                    "caption": caption
                                }
                                f.write(json.dumps(result) + "\n")  # Write each result as a JSON object followed by a newline
                                f.flush()  # Ensure the data is written to disk
                                logger.info(f"Disaster: {disaster_dir}, Image: {image_file}, Caption: {caption}")
                                processed_images += 1

                                if processed_images % 100 == 0:
                                    print(f"{processed_images}/{max_images}")

                                if processed_images >= max_images:
                                    break
                            except Exception as e:
                                logger.error(f"Failed to process image {image_file} in {disaster_dir}: {e}")
                if processed_images >= max_images:
                    break
    except Exception as e:
        logger.error(f"Failed to write to the file {output_file}: {e}")

    logger.info(f"Captioning completed. Results saved to {output_file}")

def main():
    image_directory = "../Dataset/xBD/png_disaster_sorted"
    output_file = "./out/captions.jsonl"
    logger = setup_logging()
    logger.info("Starting the image captioning process...")
    process_images(image_directory, output_file, logger, max_images=UNLIMITED_NUM)

if __name__ == "__main__":
    main()
