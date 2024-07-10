# Note for Installation

```bash
# Create the conda environment and install base dependencies
conda create -n sd2 python=3.8.5 pip=20.3 cudatoolkit=11.3 pytorch=1.12.1 torchvision=0.13.1 numpy=1.23.1 -c pytorch -c defaults

# Activate the environment
conda activate sd2

# Install the additional packages using pip
pip install albumentations==1.3.0 opencv-python==4.6.0.66 imageio==2.9.0 imageio-ffmpeg==0.4.2 pytorch-lightning==1.4.2 omegaconf==2.1.1 test-tube>=0.7.5 streamlit==1.12.1 einops==0.3.0 transformers==4.19.2 webdataset==0.2.5 kornia==0.6 open_clip_torch==2.0.2 invisible-watermark>=0.1.5 streamlit-drawable-canvas==0.8.0 torchmetrics==0.6.0 -e .

pip install xformers==0.0.16

conda install -c conda-forge conda-pack

conda pack -n sd2 -o sd2.tar.gz

mv sd2.tar.gz /mnt/d

tar -xzf sd2.tar.gz -C /home/gis2024/.conda/envs/sd2
```
