# ./test/sd2_text_image.py
import sys
sys.path.append('/home/gis2024/local/Group1/SD-All/library/')
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

model_path = "./models/stabilityai/stable-diffusion-2-1"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "High quality photo of an astronaut riding a horse in space"
image = pipe(prompt, num_inference_steps=10).images[0]
image.save("./out/sd2_example.png")
