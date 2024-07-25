# scripts/ablation_size_i2i.py

import os
import logging
import yaml
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from image_display import display_ablation_size_i2i
from utils import setup_logging


def main():

    # Configure logging
    with open("configs/config_size_i2i.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    setup_logging(config["logging"]["log_file"])

    models = config["models"]
    output_folder = config["data"]["output_folder"]
    ablation_config = config["ablation_config"]

    os.makedirs(output_folder, exist_ok=True)

    display_ablation_size_i2i(
        models,
        config["ablation_config"]["seed"],
        config["data"]["ref_image"],
        ablation_config["prompt"],
        ablation_config["inference_steps"],
        ablation_config["height"],
        ablation_config["width"],
        ablation_config["guidance_scale"],
        output_folder,
    )


if __name__ == "__main__":
    main()
