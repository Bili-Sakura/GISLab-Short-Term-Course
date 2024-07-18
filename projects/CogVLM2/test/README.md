# Basic Demo

> Source: https://github.com/THUDM/CogVLM2/edit/main/basic_demo/README.md

### Minimum Requirements

Python: 3.10.12 or above

OS: It is recommended to run on a Linux operating system with NVIDIA GPU to avoid installation issues with
the `xformers` library.

GPU requirements are as shown in the table below:

| Model Name                                 | 19B Series Model                          | Remarks                       |
|--------------------------------------------|-------------------------------------------|-------------------------------|
| BF16 inference                      | 42GB                                      | Tested with 2K dialogue text  |
| Int4 inference                             | 16GB                                      | Tested with 2K dialogue text  |
| BF16 Lora Tuning (With Vision Expert Part) | 73GB(8 GPUs with A100 x 80G using zero 2) | Trained with 2K dialogue text |

Before running any code, make sure you have all dependencies installed. You can install all dependency packages with the
following command:

```shell
pip install -r requirements.txt
```

## Using CLI Demo

Run this code to start a conversation at the command line. Please note that the model must be loaded on a GPU

```shell
CUDA_VISIBLE_DEVICES=0 python cli_demo.py
```

If you want to use `int4` (or `int8`) quantization, please use 
```shell
CUDA_VISIBLE_DEVICES=0 python cli_demo.py --quant 4
```

If you have multiple GPUs, you can use the following code to perform multiple pull-up models and distribute different
layers of the model on different GPUs.

```shell
python cli_demo_multi_gpus.py
```

In `cli_demo_multi_gpus.py`, we use the `infer_auto_device_map` function to automatically allocate different layers of
the model to different GPUs. You need to set the `max_memory` parameter to specify the maximum memory for each GPU. For
example, if you have two GPUs, each with 23GiB of memory, you can set it like this:

```python
device_map = infer_auto_device_map(
    model=model,
    max_memory={i: "23GiB" for i in range(torch.cuda.device_count())},
    # set 23GiB for each GPU, depends on your GPU memory, you can adjust this value
    no_split_module_classes=["CogVLMDecoderLayer"]
)
```

