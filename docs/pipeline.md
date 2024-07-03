# Pipeline of Short-Term Course Project: Post-event Remote Sensing Image Generation based on Conditional Diffusion Model

## Overview

- Develop a Conditional Diffusion Model specialized for post-event remote sensing image generation
- Compare our model with mainstream model to show the strength (e.g. [FID](https://proceedings.neurips.cc/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html), downstream task augmentation, less hallucination) of our model 
- Produce a synthetic dataset based our model  

> Future Work:
>
> 1. ControlNet++: Improving Conditional Controls with Efficient Consistency Feedback
>    - [Code](https://github.com/liming-ai/ControlNet_Plus_Plus)
>    - [Paper (Accepted by ECCV 2024)](https://liming-ai.github.io/ControlNet_Plus_Plus/)
> 2. Uni-ControlNet: All-in-One Control to Text-to-Image Diffusion Models
>    - [Code](https://github.com/ShihaoZhaoZSH/Uni-ControlNet)
>    - [Paper (Accepted by NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2468f84a13ff8bb6767a67518fb596eb-Abstract-Conference.html)
> 3. OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping
>    - [Dataset](https://open-earth-map.org/)
>    - [Paper (Accepted by WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/html/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.html)

## Experiment Pipeline

1. Generate (pre/post-event) Remote Sensing Images with [Stable Diffusion](https://github.com/CompVis/stable-diffusion) (prompt only)

2. Generate (pre/post-event)  Remote Sensing Images with [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat) (no conditioning version)

   - prompt only
   - prompt & reference image (i.e. pre-event image)

   > Compared with step1, DiffusionSAT generates better post-event remote sensing images (RSIs).

3. Generate (pre/post-event) Remote Sensing Images with [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat) (conditioning version)

   - prompt only
   - prompt & reference image
   - prompt & reference image & segmentation map

   > Compared with step2, DiffusionSAT with conditioning fine-tuning (i.e. [ControlNet](https://github.com/lllyasviel/ControlNet)) generates better post-event RSIs.

4. Further Fine-tuning DiffusionSAT, focus on the post-event RSIs Generation Capabilty

   - [xBD](https://xview2.org/dataset) (satellite: WorldView/QuickBird, resolution: ~0.5m)
   - `Vivid Standard Imagery Basemaps` samples from [Maxar](https://resources.maxar.com/product-samples/vivid-standard-imagery-basemaps-global-locations) (satellite: WorldView/QuickBird,  resolution: ~0.5m)
   - ... (ref to [heating dataset](../paper_writing/paper.md#dataset))