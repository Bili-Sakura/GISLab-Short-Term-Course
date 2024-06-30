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

> 1. **CrowdAI Mapping Challenge Dataset (2018)**  
>    Mohanty, S. P. (2018). CrowdAI mapping challenge dataset. Available: [https://www.aicrowd.com/challenges/mapping-challenge/dataset_files](https://www.aicrowd.com/challenges/mapping-challenge/dataset_files).
>
> 2. **Massachusetts Buildings Dataset (2013)**  
>    Mnih, V. (2013). *Machine Learning for Aerial Image Labeling*. Toronto, ON, Canada: University of Toronto.
>
> 3. **Open AI Dataset (2018)**  
>    Ji, S., Wei, S., & Lu, M. (2019). Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set. *IEEE Transactions on Geoscience and Remote Sensing*, 57(1), 574–586. doi:10.1109/TGRS.2018.2858817.
>
> 4. **WHU Building Dataset (2018)**  
>    Ji, S., Wei, S., & Lu, M. (2020). Toward automatic building footprint delineation from aerial images using CNN and regularization. *IEEE Transactions on Geoscience and Remote Sensing*, 58(3), 2178–2189. doi:10.1109/TGRS.2019.2944422.
>
> 5. **SpaceNet Challenge Dataset (2017)**  
>    Zhao, W., Persello, C., & Stein, A. (2021). Building outline delineation: From aerial images to polygons with an improved end-to-end learning framework. *ISPRS Journal of Photogrammetry and Remote Sensing*, 175, 119–131. doi:10.1016/j.isprsjprs.2021.02.014.
>
> 6. **Learning Aerial Image Segmentation From Online Maps Dataset (2017)**  
>    Kaiser, P., Wegner, J. D., Lucchi, A., Jaggi, M., Hofmann, T., & Schindler, K. (2017). Learning aerial image segmentation from online maps. *IEEE Transactions on Geoscience and Remote Sensing*, 55(11), 6054–6068. doi:10.1109/TGRS.2017.2719863.
>
> 7. **Inria Aerial Image Labeling Dataset (2017)**  
>    Maggiori, E., Tarabalka, Y., Charpiat, G., & Alliez, P. (2017). Can semantic labeling methods generalize to any city? The Inria aerial image labeling benchmark. In *Proceedings of the IEEE International Geoscience and Remote Sensing Symposium* (pp. 3226–3229). doi:10.1109/IGARSS.2017.8127684.
>
> 8. **WHU-Mix Building Dataset**  
>    Luo, M., Ji, S., & Wei, S. (2023). A Diverse Large-Scale Building Dataset and a Novel Plug-and-Play Domain Generalization Method for Building Extraction. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 16, 4122–4138. doi:10.1109/JSTARS.2023.3268176.
>

<details>
<summary>Latex</summary>
\begin{table}[h!]
\centering
\begin{tabular}{|l|l|r|r|r|l|}
\hline
\textbf{Dataset} & \textbf{Subsets} & \textbf{Number of image tiles} & \textbf{Image size} & \textbf{Image spatial resolution (m)} & \textbf{Area (km\textsuperscript{2})} & \textbf{Image type} \\
\hline
CrowdAI mapping challenge dataset (2018) & / & 341058 & 300 × 300 & unknown & / & satellite \\
\hline
Massachusetts buildings dataset (2013) & / & 151 & 1500 × 1500 & 1 & 340 & aerial \\
\hline
Open AI dataset (2018) & / & 13 & $\sim$40 000 × 40 000 & $\sim$0.07 & 102 & aerial \\
\hline
\multirow{2}{*}{WHU building dataset (2018)} & Aerial imagery dataset & 8188 & 512 × 512 & 0.3 & 193 & aerial \\
 & Satellite dataset I (global cities) & 204 & 512 × 512 & 0.3--2.5 & $\sim$5 & satellite \\
 & Satellite dataset II (East Asia) & 17388 & 512 × 512 & 0.35 & 558 & satellite \\
\hline
\multirow{5}{*}{SpaceNet challenge dataset (2017)} & Rio de Janeiro & 6940 & 438 × 406 & 0.5 & 308 & satellite \\
 & Vegas & 3851 & 438 × 406 & 0.5 & 146 & satellite \\
 & Paris & 1148 & 438 × 406 & 0.5 & 44 & satellite \\
 & Shanghai & 4582 & 650 × 650 & 0.3 & 174 & satellite \\
 & Khartoum & 1012 & 438 × 406 & 0.5 & 38 & satellite \\
\hline
\multirow{5}{*}{Learning Aerial Image Segmentation From Online Maps dataset (2017)} & Berlin & 200 & $\sim$3000 × 3000 & $\sim$0.1 & 10 & aerial \\
 & Chicago & 497 & $\sim$3000 × 3000 & $\sim$0.1 & 51 & aerial \\
 & Paris & 625 & $\sim$3000 × 3000 & $\sim$0.1 & 60 & aerial \\
 & Potsdam & 24 & $\sim$3000 × 3000 & $\sim$0.1 & 2 & aerial \\
 & Zurich & 375 & $\sim$3000 × 3000 & $\sim$0.1 & 36 & aerial \\
\hline
Inria aerial image labeling dataset (2017) & Training set (Austin, Chicago, Kitsap County, Western Tyrol, Vienna) & 180 & 5000 × 5000 & 0.3 & 405 & aerial \\
\hline
\multirow{2}{*}{WHU-Mix building dataset} & Trainval set & 43727 & 512 × 512 & 0.091--2.5 & 1080 & aerial/satellite \\
 & Test set & 7718 & 512 × 512 & 0.091--2.5 & 133 & aerial/satellite \\
\hline
\end{tabular}
\caption{Comparison of different building datasets}
\label{table:1}
    \end{table}</details>

### References
