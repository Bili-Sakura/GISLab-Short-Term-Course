import os
import torch
from diffusionsat import (
    DiffusionSatPipeline,
    DiffusionSatControlNetPipeline,
    SatUNet,
    ControlNetModel3D,
)
from src.image_utils import load_image

def generate_image_with_diffusion_pipeline(prompt, model_path, metadata, num_inference_steps, guidance_scale, height, width):
    unet = SatUNet.from_pretrained(os.path.join(model_path, 'checkpoint-150000'), subfolder="unet", torch_dtype=torch.float32)
    pipeline = DiffusionSatPipeline.from_pretrained(model_path, unet=unet, torch_dtype=torch.float32)
    pipeline = pipeline.to("cuda")

    image = pipeline(prompt, metadata=metadata, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width).images[0]
    return image

def generate_image_with_controlnet(prompt, control_image_path, model_path, controlnet_path, metadata, num_inference_steps, guidance_scale, height, width):
    control_image = load_image(control_image_path)

    controlnet = ControlNetModel3D.from_pretrained(os.path.join(controlnet_path, 'checkpoint-50000'), subfolder="controlnet", torch_dtype=torch.float32)
    unet = SatUNet.from_pretrained(model_path, subfolder="unet", torch_dtype=torch.float32)
    pipeline = DiffusionSatControlNetPipeline.from_pretrained(
        model_path, unet=unet, controlnet=controlnet, torch_dtype=torch.float32
    )
    pipeline = pipeline.to("cuda")

    image = pipeline(prompt, image=control_image, metadata=metadata, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width, is_temporal=True).images[0]
    return image
