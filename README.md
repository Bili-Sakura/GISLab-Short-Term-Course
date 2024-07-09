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

- [x] Read the work of [DiffusionSAT](readings\RS\Khanna et al_2023_DiffusionSat.pdf) carefully on `research background`, `pretraining and fine-tuning dataset` ,`model architecture` and `evaluation` （see [notes](./discussion.md#76sakura)）

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

- [ ] Set up the environment of  [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat), [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) respectively
- [ ] Read the paper and code of [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat)
- [x] Further design the pipeline of our experiments (see [here](./docs/pipeline.md/#experiments))
- [ ] Think about how to generate batch of prompts with language model assistant.

### July<sup>9th</sup>

Problems and Solutions:

- [ ] None

To-do List:

- [ ] Set up the environment of  [DiffusionSAT](https://github.com/samar-khanna/DiffusionSat), [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and [ControlNet](https://github.com/lllyasviel/ControlNet) respectively
- [ ] Read the paper and code of `DiffusionSAT`, see materials [here](./docs/introduction_to_diffusionsat.md)
- [ ] Think about how to generate batch of prompts with language model assistant.
