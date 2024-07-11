# Manuscript

> Overleaf Project: [Link](<https://cn.overleaf.com/project/6657330377cf39cf52e9d451>)

## GBD: A Global-Scale Building Damage Dataset based on Diffusion Model

**Abstrct**: Object Detection of Damaged Buildings in disastrous events is important for aiding and reconstruction. Current approaches for building damage assessment include CNN-based models and transformer-based models. However, these pre-trained models lack general capability and fail in timeliness of detection in terms of disastrous events. In this work, we propose a generative model, ModelName, to manufacture potential post-disaster images from vulnerable regions on a global scale, named GBD. We find that after further training state-of-the-art models on GBD, the performance of models shows great improvements.

**Index Terms**——Object detection, building damage assessment, post-event, pre-trained model, generative model

### Introduction

> Earth Observation

> Disater Monitor

> Traditional Methods and their Limitations

> Our Method and Contribution

### Related Work

### Methodology

### Experienments and Discussions

#### Dataset

| Dataset                                                  | Subsets                                      | Number of image tiles | Image size     | Image spatial resolution (m) | Area (km²) | Image type    |
|----------------------------------------------------------|----------------------------------------------|-----------------------|----------------|------------------------------|------------|---------------|
| CrowdAI mapping challenge dataset (2018)                 | /                                            | 341058                | 300 × 300      | unknown                      | /          | satellite     |
| Massachusetts buildings dataset (2013)                   | /                                            | 151                   | 1500 × 1500    | 1                            | 340        | aerial        |
| Open AI dataset (2018)                                   | /                                            | 13                    | ~40 000 × 40 000 | ~0.07                        | 102        | aerial        |
| WHU building dataset (2018)                              | Aerial imagery dataset                       | 8188                  | 512 × 512      | 0.3                          | 193        | aerial        |
|                                                          | Satellite dataset I (global cities)          | 204                   | 512 × 512      | 0.3–2.5                      | ~5         | satellite     |
|                                                          | Satellite dataset II (East Asia)             | 17388                 | 512 × 512      | 0.35                         | 558        | satellite     |
| SpaceNet challenge dataset (2017)                        | Rio de Janeiro                               | 6940                  | 438 × 406      | 0.5                          | 308        | satellite     |
|                                                          | Vegas                                        | 3851                  | 438 × 406      | 0.5                          | 146        | satellite     |
|                                                          | Paris                                        | 1148                  | 438 × 406      | 0.5                          | 44         | satellite     |
|                                                          | Shanghai                                     | 4582                  | 650 × 650      | 0.3                          | 174        | satellite     |
|                                                          | Khartoum                                     | 1012                  | 438 × 406      | 0.5                          | 38         | satellite     |
| Learning Aerial Image Segmentation From Online Maps dataset (2017) | Berlin                                       | 200                   | ~3000 × 3000   | ~0.1                         | 10         | aerial        |
|                                                          | Chicago                                      | 497                   | ~3000 × 3000   | ~0.1                         | 51         | aerial        |
|                                                          | Paris                                        | 625                   | ~3000 × 3000   | ~0.1                         | 60         | aerial        |
|                                                          | Potsdam                                      | 24                    | ~3000 × 3000   | ~0.1                         | 2          | aerial        |
|                                                          | Zurich                                       | 375                   | ~3000 × 3000   | ~0.1                         | 36         | aerial        |
| Inria aerial image labeling dataset (2017)               | Training set (Austin, Chicago, Kitsap County, Western Tyrol, Vienna) | 180                   | 5000 × 5000    | 0.3                          | 405        | aerial        |
| WHU-Mix building dataset                                 | Trainval set                                 | 43727                 | 512 × 512      | 0.091–2.5                    | 1080       | aerial/satellite |
|                                                          | Test set                                     | 7718                  | 512 × 512      | 0.091–2.5                    | 133        | aerial/satellite |


### References
