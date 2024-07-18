# CogVLM 2 Tutorial

> Author: Sakura  
> Last Update: 18 July, 2024


## Quick Start

```bash
cd cogvlm
conda activate sd-all-latest # pr `sd-all-0716`

# for test; We have 4 NVIDIA 3090 24G on server, as GPU2 is occupied by other group, we use other GPUs.
CUDA_VISIBLE_DEVICES=3,1,0 python test/cli_demo_multi_gpus.py
# test int4 version, which only require 16G Memory
CUDA_VISIBLE_DEVICES=0 python test/cli_demo.py --quant 4
```