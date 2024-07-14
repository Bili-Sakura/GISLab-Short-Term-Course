import argparse
import logging
import math
import os
import random
import webdataset as wds
from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
from PIL import Image

from pyproj import Geod
from shapely.geometry import shape as shapey
from shapely.wkt import loads as shape_loads

import torch
from torchvision import transforms

import sys
import os

sys.path.append("/home/gis2024/local/Group1/SD-All/library/")

from diffusionsat import (
    SatUNet,
    DiffusionSatPipeline,
    SampleEqually,
    fmow_tokenize_caption,
    fmow_numerical_metadata,
    spacenet_tokenize_caption,
    spacenet_numerical_metadata,
    satlas_tokenize_caption,
    satlas_numerical_metadata,
    combine_text_and_metadata,
    metadata_normalize,
)

# cache location
# os.environ["HF_HOME"] = "path/to/.cache/"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt",
    type=str,
    default="An urban satellite image highlighting green spaces in Central Park, New York City",
)
parser.add_argument(
    "--metadata",
    nargs="+",
    type=float,
    default=[-73.968285, 40.785091, 0.920, 0.080, 2022, 5, 10],
)
parser.add_argument("--output_path", type=str, default="newyork.png")
args = parser.parse_args()

# Or provide metadata values and then normalize
caption = args.prompt
metadata = metadata_normalize(args.metadata).tolist()

path = "./models/fsx/proj-satdiffusion/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/"
unet = SatUNet.from_pretrained(
    path + "checkpoint-150000", subfolder="unet", torch_dtype=torch.float32
)
# unet_ema_diffusion_pytorch_model-001.bin -> diffusion_pytorch_model.bin
# unet = SatUNet.from_pretrained(path , subfolder="unet", low_cpu_mem_usage=False, device_map=None, torch_dtype=torch.float16)
pipe = DiffusionSatPipeline.from_pretrained(path, unet=unet, torch_dtype=torch.float32)
pipe = pipe.to("cuda")

image = pipe(
    caption,
    metadata=metadata,
    num_inference_steps=20,
    guidance_scale=7.5,
    height=512,
    width=512,
).images[0]
image.save(args.output_path)
