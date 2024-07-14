# Deploy Stable Diffusion from Scratch with Diffusers

> Author: Sakura
>
> Last Updated:14 June,2024

## Outline

- Quick Start

  - Setup a simple project

  - Download model weights

  - Setup environment

  - Generate your first image with diffusers' pipeline

- Fine-tuning your own Stable Diffusion Model

  - Prepare training dataset
  - Further training
    - LORA
  - ControlNet

- Common Q&A

  - The model file structure and how they work
  - What's difference between model weights format (e.g. *.safetensors,*.bin, *.ckpt)

## Quick Start

### Setup a simple project

As we are using [diffusers](https://hf-mirror.com/docs/diffusers/v0.29.2/en) package for stable diffusion based model deployment, there is **little** code required a simple text-to-image/image-to-image generation project. Just create a project in the most simpliest way as below.

```bash
SD-Difusers-all-in-one/
├── data/
├── src/
├── main.py
├── .gitignore
├── README.md
└──requirements.txt
```

### Download model weights

These heating models/projects always have a huggingface repo contains all you need. Here, we take [Stable Diffusion v2.1](https://hf-mirror.com/stabilityai/stable-diffusion-2-1) as an example.

> As [huggingface.co](https://huggingface.co/) often encounters network error, we highly recommend to use mirror webpage namely [hf-mirror](https://hf-mirror.com/).

1. Look for the Official Repo for [Stable Diffusion v2.1](https://hf-mirror.com/stabilityai/stable-diffusion-2-1)

<img src="C:/Users/Administrator/AppData/Roaming/Typora/typora-user-images/image-20240714112716028.png" alt="image-20240714112716028" style="zoom:50%;" />

2. Go for `Files and versions`

![image-20240714113002502](../assets/diffusers_from_scratch/sd21_repo_file_page.png)

You can get these files in 3 ways.

A. Install `diffusers` python package and downloading model file with well-organized structure into your project as follows:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1")
```

> [!NOTE]
> Before doing this, you need have access to huggingface token, set it through huggingface-cli and check connection to [huggingface.co](https://huggingface.co/), because `from_pretrained` function would pull the model from  [huggingface.co](https://huggingface.co/) with corresponding model hub page.

B. Clone this repo using git clone as well as **git-lfs** (git for large files)

```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://hf-mirror.com/stabilityai/stable-diffusion-2-1
# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://hf-mirror.com/stabilityai/stable-diffusion-2-1
```

> Still, you need have access to huggingface token.

C. Manually download by multi-click (the way the author use)

As there would not be too many files for downloading, manually downloading files is okay.

> [!CAUTION]
> You should manually correct the file name when downloading file from a subfolder in root directory. For instance, a file named `scheduler_config.json` under `scheduler` folder would turn to `scheduler_scheduler_config.json` (being added folder name with '_'). 
>
> ```bash
> root/ #huggingface hub 
> ├── scheduler/
> 	└──scheduler_config.json
> ```
>
> ```bash
> SD-Difusers-all-in-one/ #your project 
> ├── models/
> |   └──scheduler/
> |		└──scheduler_scheduler_config.json # should be correct into scheduler_config.json
> ├── data/
> ├── src/
> ├── main.py
> ├── .gitignore
> ├── README.md
> └──requirements.txt
> 
> ```

### Setup environment

For quick start, only a few packages is required (but can be large). If you want to deploy with GUI, you can refer to the requirements.txt in correspondent  GitHub Repo.

````python
# requirements.txt

## common conda packages

## Required Package:
diffusers
transformers
pytorch
accelerate
xformer

## Optional Packages:
streamlit # GUI
gradio # GUI
matplotlib # ....
````

> [!CAUTION]
>
> The `diffusers` package is updating fast. Therefore, we always manually download source code from https://github.com/huggingface/diffusers, and place it under `/src` or `/lib` in root directory. Given this, we would import the `diffusers` package as follow:
>
> ```bash
> from lib.diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
> # instead of 'from diffusers import ...' which import the module from conda env