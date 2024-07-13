import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

pipe = StableDiffusion3Img2ImgPipeline.from_pretrained("../checkpoint", torch_dtype=torch.float16, text_encoder_3=None,tokenizer_3=None)
pipe = pipe.to("cuda")
init_image = load_image("India_pre.png")
prompt = "A satellite image of Indian farmland suffering from floods, on a small scale"
image = pipe(prompt, image=init_image).images[0]
image.save("output_India_18_SD_reference.png")
'''
init_image = load_image("image.png")
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipe(prompt, image=init_image).images[0]
image.save("image_to_image.png")
'''