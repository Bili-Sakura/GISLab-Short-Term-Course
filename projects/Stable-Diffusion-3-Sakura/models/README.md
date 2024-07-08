# Details for Model Files

## Directory Structure

Model files are download from [HuggingFace/Stability-AI](https://huggingface.co/stabilityai/stable-diffusion-3-medium)

```arduino
models/
├── clip_g.safetensors (only one version)
├── clip_l.safetensors (only one version)
├── t5xxl.safetensors   (renamed from `t5xxl_fp16.safetensors`, only one version)
└── sd3_medium.safetensors  (renamed from `sd_medium_incl_clips_t5xxlfp16.safetensors`, largest/best version out of 4 version)
```

## Notes

1. We do not use separate `VAE` (which is optional) in our preject.
2. T5-XXL model (from Google) and SD3 model (from Stability AI) being used in our project are the best/largest model weight files.
3. CLIP-G and CLIP-L (from OpenAI) are both used in SD3.

- CLIP-G: Larger, more powerful model suitable for detailed and complex tasks, with higher hidden size, intermediate size, number of attention heads, and layers.
- CLIP-L: More lightweight model designed for efficiency, with lower hidden size, intermediate size, number of attention heads, and layers, but still highly effective for many applications.
