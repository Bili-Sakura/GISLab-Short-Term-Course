# ./test/sd3_text_to_image.py
import sys
sys.path.append('/home/gis2024/local/Group1/SD-All/library/')
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "./models/stabilityai/stable-diffusion-3-medium-diffusers", 
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None
    )
pipe.to("cuda")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=10,
    height=512,
    width=512,
    guidance_scale=7.0,
).images[0]

image.save("./out/sd3_example.png")