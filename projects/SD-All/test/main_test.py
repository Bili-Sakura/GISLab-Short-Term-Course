import sys
import json

# from pathlib import Path
sys.path.append("/home/gis2024/local/Group1/SD-All/library/")
# sys.path.remove("/home/gis2024/.conda/envs/sd-all-0715v3/lib/python3.10/site-packages")
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoPipelineForText2Image,
    StableDiffusion3Pipeline,
)
import torch


def generate_image(
    model_name: str,
    prompt: str,
    output_path: str,
    num_inference_steps: int = 10,
    guidance_scale: float = 7.0,
    height: int = 512,
    width: int = 512,
):
    if model_name == "sd2":
        model_path = "./models/stabilityai/stable-diffusion-2-1"
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to("cuda")

        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        image.save(output_path)

    elif model_name == "sdxl_turbo":
        pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            "./models/stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        )
        pipeline_text2image = pipeline_text2image.to("cuda")

        image = pipeline_text2image(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]
        image.save(output_path)

    elif model_name == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "./models/stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None,
        )
        pipe.to("cuda")

        image = pipe(
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
        ).images[0]
        image.save(output_path)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


def generate_images_from_config(config_path: str):
    with open(config_path, "r") as file:
        configs = json.load(file)

    for config in configs:
        generate_image(
            model_name=config["model_name"],
            prompt=config["prompt"],
            output_path=config["output_path"],
            num_inference_steps=config.get("num_inference_steps", 10),
            guidance_scale=config.get("guidance_scale", 7.0),
            height=config.get("height", 512),
            width=config.get("width", 512),
        )


if __name__ == "__main__":
    config_path = "./data/samples.json"
    generate_images_from_config(config_path)
