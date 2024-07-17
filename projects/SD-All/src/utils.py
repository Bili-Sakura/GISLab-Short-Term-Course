# src/utils.py

def get_model_id(model_option: str) -> str:
    model_mapping = {
        "sd2": "./models/stabilityai/stable-diffusion-2-1",
        "sdxl": "./models/stabilityai/sdxl-turbo",
        "sd3": "./models/stabilityai/stable-diffusion-3-medium-diffusers",
        "diffusionsat": "./models/fsx/proj-satdiffusion/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/"
    }
    return model_mapping.get(model_option, "./models/stabilityai/stable-diffusion-2-1")  # Default to sd2 if not specified
