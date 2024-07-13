import torch
from lib.diffusers import StableDiffusion3Pipeline

# 24GB VRAM, clip_g + clip_l + T5_xxl
pipe = StableDiffusion3Pipeline.from_pretrained(
    "../checkpoint", torch_dtype=torch.float16, text_encoder_3=None, tokenizer_3=None
)
pipe = pipe.to("cuda")

# generate in 1024*1024
image = pipe(
    "A cat holding a sign that says Hello SD3",
    negative_prompt="",
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]

# save
image.save("image.png")
