# ./test/sd2_text_image.py
import sys
sys.path.append('/home/gis2024/local/Group1/SD-All/library/')
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
from compel import Compel

model_path = "./models/stabilityai/stable-diffusion-2-1"
pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
prompt = "A satellite image of Indian farmland suffering from floods, with one building in the image"
prompt_embeds = compel_proc(prompt)
generator = torch.manual_seed(3)

image = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=10).images[0]
image.save("./out/sd2_weighting_1.png")
