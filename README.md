# GISLab_Course

"Introduction to Diffusion Models" - GIS Lab 2024 Short-term Course

See online keynote(.md) [here](./docs/keynote.md).

Keynote(.pptx) download link [here](https://pan.baidu.com/s/1NAZi_NWV3lNLi1rNXhJxhA?pwd=0702).

## Agenda & Discussion Board

### July<sup>3rd</sup>

To-do List:

- [x] GitHub Collaboration & Overleaf Collaboration (optional)
- [x] Understand General Project Purpose and Pipeline (see [here](./docs/pipeline.md))

### July<sup>4th</sup>

Problems and Solutions:

- [x] Cuda Out of Memory (details see [here](./discussion.md#74-syt))
  1. Check whether already use GPU, if true, turn to `step2`.
  2. Compared the total memory of your GPU and the desired memory of the project. If the desired one exceeds the local capability, turn `step3`.
  3. Deploy the project on the GPU clusters from GISLab. First, make sure you are confident with what you are going to do with this project as well as the experiments in mind. Then, I will ask the administrator to register the cluster for me. Finally, regarding the deployment on a cluster server, ask the TA. Liang for details.

To-do List:

- [x] Prepare Presentation on `Project Understanding and Experiment Design` for the next day (July 5th)

### July<sup>5th</sup>

Problems and Solutions:

- [x] Suggestions given by Prof. Zhang

1. Estimate the model performance for current models ✔

2. Pay attention to labeled data ✔

To-do List:

- [x] Get to know the heating dataset (e.g. [xBD](https://xview2.org/dataset))

### July<sup>6th</sup>

Problems and Solutions:

- [x] Network Error with Server Cluster

To-do List:

- [x] Read the work of [DiffusionSAT](./readings/DiffusionSAT/Khanna%20et%20al_2023_DiffusionSat.pdf) carefully on `research background`, `pretraining and fine-tuning dataset` ,`model architecture` and `evaluation` （see [notes](./discussion.md#76sakura)）

### July<sup>7th</sup>

Problems and Solutions:

- [x] None

To-do List:

- [x] Prepare Presentation for the next day (July 8th)
  - display samples generate by Stable Diffusion
  - display samples generate by DiffusionSAT
  - compare their difference with same prompt in naive way
  - discuss the future evaluation **metrics**

### July<sup>8th</sup>

Problems and Solutions:

- [x] None

To-do List:

- [x] Further design the pipeline of our experiments (see [here](./docs/pipeline.md/#experiments))
- [ ] Set up the environment of  [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat), [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) respectively
- [ ] Read the paper and code of [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat)
- [ ] Think about how to generate batch of prompts with language model assistant.

### July<sup>9th</sup>

Problems and Solutions:

- [x] None

To-do List:

- [x] Read the paper and code of `DiffusionSAT`, see materials [here](./docs/introduction_to_diffusionsat.md) along with [PowerPoint](./lectures/Introduction_to_DiffusionSAT.pptx)
- [ ] Set up the environment of  [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat), [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) respectively
  
- [ ] Think about how to generate batch of prompts with language model assistant.

Notes:

1. xBD has 2 version of publication, be it [CVPR Workshop 2018](https://openaccess.thecvf.com/content_CVPRW_2019/html/cv4gc/Gupta_Creating_xBD_A_Dataset_for_Assessing_Building_Damage_from_Satellite_CVPRW_2019_paper.html) and [arXiv 2019](https://arxiv.org/abs/1911.09296), the arXiv version is correspondent to the released xBD dataset, where CVPR version is out-of-date
2. The evolution of Stable Diffusion refers to note [here](./discussion.md#introduction-to-stable-diffusion-series-model). Stable Diffusion v1,v1.x,v2.1 (2021-2022) have same architecture which is used for ControlNet and  DiffusionSat. Stable Diffusion XL and XL Turbo are released in 2023 with new architecture. Stable Diffusion 3 is released in 2024.6 which is pretrained on 1B images and has a rectified workflow architecture that combined much more encoder modules than before.

### July<sup>10th</sup>-July<sup>13th</sup>

Problems and Solutions:

- [x] None

To-do List:

- [x] Generate batch of prompts with language model assistant - [RSGPT](./projects/RSGPT/README.md).
- [x] Deploy Stable Diffusion 3 on Server which is capable of image-to-image generation
- [x] DiffusionSat is able to use .json file for batch generation (test-version)
- [ ] Try [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) for image captioning
- [ ] Read documentation of [diffusers](https://hf-mirror.com/docs/diffusers/v0.29.2/en/index) package
- [ ] Prepare dataset for training ControlNet in DiffusionSat, follow [ControlNet training example by diffusers](./docs/train_with_diffusers.md) to train diffusionsat that is able to execute in painting task (i.e. generated post-event image with given pre-event image as shown in [paper](./readings/DiffusionSAT/Khanna%20et%20al_2023_DiffusionSat.pdf))

### July<sup>14th</sup>

To-do List:

- [x] Using `diffusers` pipeline unified all projects into `SD-All` including SD v2.1, SD XL Turbo, SD 3 and DiffusionSat
- [ ] ...
