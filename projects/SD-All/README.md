# Stable Diffusion with Diffusers

> Author: Sakura  
> Last Update: 14 July, 2024

This project use [diffusers](https://github.com/huggingface/diffusers), which is a well-organized pipeline host by huggingface for deployment of diffusion models.

## Quick Start

Using Stable Diffusion v2.1 for text-to-image Generation

```bash
conda activate sd-all
python ./test/sd2_text_image.py
```

```python
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
image = pipe(prompt, num_inference_steps=25).images[0]
image.save("output.png")

```
