import argparse
import os
import torch
from src.config_processor import process_config_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default="./input")
    parser.add_argument('--output_folder', type=str, default="./out")
    # parser.add_argument('--generator',type=torch.Generator,default=None)
    seed=0
    # parser.add_argument('--generator',type=torch.Generator,default=torch.manual_seed(seed))
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    for config_file in os.listdir(input_folder):
        if config_file == 'samples_pre_to_post.json':
            config_path = os.path.join(input_folder, config_file)
            process_config_file(config_path, output_folder)

if __name__ == "__main__":
    main()
