# test/main_old.py

import json
from src.generate_old import generate_image


def main():
    with open("../data/batch.json", "r") as file:
        samples = json.load(file)

    for sample in samples:
        prompt = sample.get("prompt")
        model_name = sample.get("model_name")
        metadata = sample.get("metadata", [])
        output = sample.get("output", "output.png")

        if prompt and model_name:
            generate_image(prompt, model_name, output, metadata)
        else:
            print("Invalid sample configuration. Skipping...")


if __name__ == "__main__":
    main()
