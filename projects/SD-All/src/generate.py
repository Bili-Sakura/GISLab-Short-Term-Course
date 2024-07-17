# src/generate.py

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForText2Image, StableDiffusion3Pipeline
from PIL import Image
from .utils import get_model_id

def generate_image(prompt: str, model_name: str, output_path: str, metadata: list, num_inference_steps: int = 10, guidance_scale: float = 7.5, height: int = 512, width: int = 512):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompt_with_metadata = f"{prompt} {' '.join(map(str, metadata))}" if metadata else prompt

    if model_name == "sd2":
        model_path = get_model_id(model_name)
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)

        image = pipe(
            prompt_with_metadata, 
            num_inference_steps=num_inference_steps,
            height=height,
            width=width
            ).images[0]

    elif model_name == "sdxl":
        pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            get_model_id(model_name), torch_dtype=torch.float16, variant="fp16"
        )
        pipeline_text2image = pipeline_text2image.to(device)

        
        image = pipeline_text2image(
                prompt=prompt_with_metadata,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
            ).images[0]

    elif model_name == "sd3":
        pipe = StableDiffusion3Pipeline.from_pretrained(
            get_model_id(model_name),
            guidance_scale=guidance_scale,
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None,
        )
        pipe.to(device)

        
        image = pipe(
                prompt=prompt_with_metadata,
                negative_prompt="",
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
            ).images[0]

    elif model_name == "diffusionsat":
        model_path = get_model_id(model_name)
        from library.diffusionsat import (
            SatUNet, DiffusionSatPipeline,
            SampleEqually,
            fmow_tokenize_caption, fmow_numerical_metadata,
            spacenet_tokenize_caption, spacenet_numerical_metadata,
            satlas_tokenize_caption, satlas_numerical_metadata,
            combine_text_and_metadata, metadata_normalize,
        )
        unet = SatUNet.from_pretrained(model_path + 'checkpoint-150000', subfolder="unet", torch_dtype=torch.float32)
        pipe = DiffusionSatPipeline.from_pretrained(model_path, unet=unet,torch_dtype=torch.float32)
        pipe = pipe.to(device)

        image = pipe(
            prompt, 
            metadata=metadata,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width
            ).images[0]

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    image.save(output_path)
    print(f"Image saved at {output_path}")
