# Stable Diffusion with Diffusers

> Author: Sakura  
> Last Update: 14 July, 2024

This project use [diffusers](https://github.com/huggingface/diffusers), which is a well-organized pipeline host by huggingface for deployment of diffusion models.

## Quick Start

```bash
conda activate sd-all
# Using Stable Diffusion v2.1 for text-to-image Generation
python ./test/sd2_text_to_image.py 
# Using DiffusionSat for text-to-image Generation
python ./test/diffusionsat_text_to_image.py
# 
CUDA_VISIBLE_DEVICES=1 python scripts/ablation_steps_t2i.py
CUDA_VISIBLE_DEVICES=1 python scripts/ablation_steps_i2i.py
```

