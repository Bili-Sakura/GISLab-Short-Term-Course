# Pipeline of Short-Term Course Project: Post-event Remote Sensing Image Generation based on Conditional Diffusion Model

## Overview

- Develop a Conditional Diffusion Model specialized for post-event remote sensing image generation
- Compare our model with mainstream model to show the strength (e.g. FID, downstream task augmentation, less hallucination) of our model 
- Produce a synthetic dataset based our model  

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

   > Compared with step2, DiffusionSAT with conditioning fine-tuning (i.e. ControlNet) generates better post-event RSIs.

4. Further Fine-tuning DiffusionSAT, focus on the post-event RSIs Generation Capabilty

   - [xBD](https://xview2.org/dataset) (satellite: WorldView/QuickBird, resolution: ~0.5m)
   - `Vivid Standard Imagery Basemaps` samples from [Maxar](https://resources.maxar.com/product-samples/vivid-standard-imagery-basemaps-global-locations) (satellite: WorldView/QuickBird,  resolution: ~0.5m)
   - ... (ref to [heating dataset](../paper_writing/paper.md#dataset))