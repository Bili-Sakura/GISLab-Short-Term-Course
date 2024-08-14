import os
import logging
import torch
from diffusers import (
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    AutoPipelineForText2Image,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusion3Img2ImgPipeline,
    CycleDiffusionPipeline,
    DDIMScheduler,
)
from typing import Literal


def load_model(
    model_name,
    model_path,
    model_type=Literal["text-to-image", "image-to-image"],
    model_varient="SDEdit",
):
    try:
        if model_name == "sd2":
            if model_type == "text-to-image":
                pipe = DiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )
            elif model_type == "image-to-image" and model_varient == "SDEdit":
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )
            elif model_type == "image-to-image" and model_varient == "CycleDiffusion":
                pipe = CycleDiffusionPipeline.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )
            else:
                logging.error(
                    f"Unknown Model Type {model_type}. Model {model_name} loaded from {model_path}"
                )
        elif model_name == "sdxl":
            if model_type == "text-to-image":
                pipe = AutoPipelineForText2Image.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )
            elif model_type == "image-to-image":
                pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )
            else:
                logging.error(
                    f"Unknown Model Type {model_type}. Model {model_name} loaded from {model_path}"
                )
        # elif model_name == "sdxl-turbo":
        #     if model_type == "text-to-image":
        #         pipe = AutoPipelineForText2Image.from_pretrained(
        #             model_path, torch_dtype=torch.float32
        #         )
        #     elif model_type == "image-to-image":
        #         pip = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        #             model_path, torch_dtype=torch.float32
        #         )
        #     else:
        #         logging.error(
        #             f"Unknown Model Type {model_type}. Model {model_name} loaded from {model_path}"
        #         )
        elif model_name == "sd3":
            if model_type == "text-to-image":
                pipe = StableDiffusion3Pipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    text_encoder_3=None,
                    tokenizer_3=None,
                )
            elif model_type == "image-to-image":
                pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    text_encoder_3=None,
                    tokenizer_3=None,
                )
            else:
                logging.error(
                    f"Unknown Model Type {model_type}. Model {model_name} loaded from {model_path}"
                )
        elif model_name == "diffusionsat":
            sd2_model_path = "models/stabilityai/stable-diffusion-2-1"
            if model_type == "text-to-image":
                unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(model_path, "checkpoint-150000"),
                    subfolder="unet",
                    torch_dtype=torch.float32,
                    use_safetensors=False,
                )

                pipe = DiffusionPipeline.from_pretrained(
                    sd2_model_path,
                    unet=unet,
                    safety_checker=None,
                    torch_dtype=torch.float32,
                )
            elif model_type == "image-to-image" and model_varient == "SDEdit":
                unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(model_path, "checkpoint-150000"),
                    subfolder="unet",
                    torch_dtype=torch.float32,
                    use_safetensors=False,
                )
                pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    sd2_model_path,
                    unet=unet,
                    safety_checker=None,
                    torch_dtype=torch.float32,
                )
            elif model_type == "image-to-image" and model_varient == "CycleDiffusion":
                unet = UNet2DConditionModel.from_pretrained(
                    os.path.join(model_path, "checkpoint-150000"),
                    subfolder="unet",
                    torch_dtype=torch.float32,
                    use_safetensors=False,
                )
                pipe = CycleDiffusionPipeline.from_pretrained(
                    sd2_model_path,
                    unet=unet,
                    safety_checker=None,
                    torch_dtype=torch.float32,
                )
            else:
                logging.error(
                    f"Unknown Model Type {model_type}. Model {model_name} loaded from {model_path}"
                )
        elif model_name == "instruct-pix2pix":
            if model_type == "text-to-image":
                logging.error(
                    f"Model {model_name} loaded from {model_path} do not support {model_type}. "
                )
            elif model_type == "image-to-image":
                pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    model_path, torch_dtype=torch.float32
                )
            else:
                logging.error(
                    f"Unknown Model Type {model_type}. Model {model_name} loaded from {model_path}"
                )
        else:
            logging.error(f"Unknown Model {model_name} loaded from {model_path}")

        pipe = pipe.to("cuda")
        logging.info(f"Model {model_name} loaded successfully from {model_path}")
        return pipe
    except Exception as e:
        logging.error(f"Error loading model {model_name} from {model_path}: {e}")
        return None
