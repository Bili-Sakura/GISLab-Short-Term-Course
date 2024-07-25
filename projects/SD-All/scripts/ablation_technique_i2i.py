# scripts/ablation_technique_i2i.py

import os
import logging
import yaml
import sys


sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from image_display import display_ablation_technique_i2i
from utils import setup_logging


def main():

    # Configure logging
    with open("configs/config_technique_i2i.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    setup_logging(config["logging"]["log_file"])

    os.makedirs(config["data"]["output_folder"], exist_ok=True)

    display_ablation_technique_i2i(
        models=config["models"],
        seed=config["ablation_config"]["seed"],
        ref_image_filepath=config["data"]["ref_image"],
        prompt=config["ablation_config"]["prompt"],
        inference_steps=config["ablation_config"]["inference_steps"],
        height=config["ablation_config"]["height"],
        width=config["ablation_config"]["width"],
        guidance_scale=config["ablation_config"]["guidance_scale"],
        output_folder=config["data"]["output_folder"],
    )


if __name__ == "__main__":
    main()
