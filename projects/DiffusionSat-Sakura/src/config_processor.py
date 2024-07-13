import os
import json
from src.image_utils import combine_images,load_image
from src.pipeline_utils import generate_image_with_diffusion_pipeline, generate_image_with_controlnet
from diffusionsat import metadata_normalize

def process_config_file(config_path, output_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    model_path = config['model_path']
    controlnet_model_path = config['controlnet_model_path']
    samples = config['samples']

    images = []
    for sample in samples:
        prompt = sample['prompt']
        metadata = sample['metadata']
        metadata = metadata_normalize(metadata).tolist()
        control_image_path = sample['control_image']
        num_inference_steps = sample['num_inference_steps']
        guidance_scale = sample['guidance_scale']
        height = sample['height']
        width = sample['width']

        if control_image_path=="":
            image_diffusion = generate_image_with_diffusion_pipeline(prompt, model_path, metadata, num_inference_steps, guidance_scale, height, width)
            images.append((image_diffusion, f"Diffusion Pipeline: {num_inference_steps} steps"))
        else:
            # control_image=load_image(control_image_path)
            # print("control_image,type:"+str(type(control_image)))
            # images.append((control_image, "ref image: "+control_image_path))
            # control_image.save(os.path.join(output_path,control_image_path))


            image_controlnet = generate_image_with_controlnet(prompt, control_image_path, model_path, controlnet_model_path, metadata, num_inference_steps, guidance_scale, height, width)
            # print("image_controlnet,type:"+str(type(image_controlnet)))
            # images.append((image_controlnet, "prompt: "+prompt))
            control_image_name_with_extension = os.path.basename(control_image_path)
            control_image_name, _ = os.path.splitext(control_image_name_with_extension)
            image_controlnet.save(os.path.join(output_path,"ref_image_"+control_image_name+"_prompt_flooding"+".png"))

            # combined_image = combine_images(images, rows=2, cols=len(samples))
            # combined_output_path = os.path.join(output_path, f"{os.path.splitext(os.path.basename(config_path))[0]}_combined.png")
            # combined_image.save(combined_output_path)

            # print(f"Combined image saved to {combined_output_path}")
