# Note for Installation

## Step 1: Env

```bash
# Create the conda environment and install base dependencies
conda create -n sd2 python=3.8.5 pip=20.3 cudatoolkit=11.3 pytorch=1.12.1 torchvision=0.13.1 numpy=1.23.1 -c pytorch -c defaults

# Activate the environment
conda activate sd2

# Install the additional Python packages
pip install albumentations==1.3.0 opencv-python==4.6.0.66 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube>=0.7.5 streamlit==1.12.1 einops==0.3.0 transformers==4.19.2 webdataset==0.2.5 kornia==0.6 open_clip_torch==2.0.2 invisible-watermark>=0.1.5 streamlit-drawable-canvas==0.8.0 torchmetrics==0.6.0

pip install diffusers
pip install numpy torch tqdm
pip install xformers==0.0.16

conda install -c conda-forge conda-pack

conda pack -n sd2 -o sd2.tar.gz

mv sd2.tar.gz /mnt/d

mkdir -p /home/gis2024/.conda/envs/sd2

tar -xzf sd2.tar.gz -C /home/gis2024/.conda/envs/sd2
```

## Step 2: Download the Model Weights

1. **Download the SD2.1-v or SD2.1-base model weights:** Place the downloaded weights in an accessible directory.

## Step 3: Sample an Image

1. Run the sampling script:

   - For SD2.1-v model (768x768 resolution):

     ```sh
     python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt <path/to/768model.ckpt> --config configs/stable-diffusion/v2-inference-v.yaml --H 768 --W 768
     ```

   - For SD2.1-base model (512x512 resolution):

     ```sh
     python scripts/txt2img.py --prompt "a professional photograph of an astronaut riding a horse" --ckpt <path/to/model.ckpt> --config <path/to/config.yaml>
     ```

2. Run the img2img script:

   - Use the following command, replacing placeholders with your specific paths and parameters:

     ```sh
     python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img <path-to-img.jpg> --strength 0.8 --ckpt <path/to/model.ckpt> --config configs/stable-diffusion/v2-inference-v.yaml
     ```

   - Here’s what each parameter means:
     - `--prompt`: The text prompt describing the desired output image.
     - `--init-img`: Path to your initial reference image.
     - `--strength`: The strength of the transformation. A value between 0 and 1, where higher values result in greater changes from the reference image.
     - `--ckpt`: Path to the model checkpoint file.
     - `--config`: Path to the configuration file for the model.

## Notes

- **DDIM Sampler:** By default, the DDIM sampler is used, rendering images in 50 steps.
- **EMA Checkpoints:** Ensure `use_ema=False` in the configuration to avoid switching from non-EMA to EMA weights.
