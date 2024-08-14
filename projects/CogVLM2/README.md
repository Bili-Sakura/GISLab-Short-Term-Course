# CogVLM 2 Tutorial

> Author: Sakura  
> Last Update: 18 July, 2024


## Quick Start

```bash
cd cogvlm
conda activate sd-all-0716 

# for test; We have 4 NVIDIA 3090 24G on server, as GPU2 is occupied by other group, we use other GPUs.
CUDA_VISIBLE_DEVICES=3,1,0 python test/cli_demo_multi_gpus.py
# test int4 version, which only require 16G Memory
CUDA_VISIBLE_DEVICES=0 python test/cli_demo.py --quant 4
```

```bash

CUDA_VISIBLE_DEVICES=3,1,0 nohup python main.py > ./log/output.log 2>&1 & # This command will start main.py in the background, redirecting both stdout and stderr to output.log
# View the output:
tail -f ./log/output.log # This command will show the last few lines of output.log and continuously update as new lines are added, effectively allowing you to monitor the output in real-time.
# stop
ps aux | grep main.py
kill -9 <PID>
```

## Prompt

> Provided by [OpenAI](https://cookbook.openai.com/examples/batch_processing)

```python
caption_system_prompt = '''
Your goal is to generate short, descriptive captions for images of items.
You will be provided with an item image and the name of that item and you will output a caption that captures the most important information about the item.
If there are multiple items depicted, refer to the name provided to understand which item you should describe.
Your generated caption should be short (1 sentence), and include only the most important information about the item.
The most important information could be: the type of item, the style (if mentioned), the material or color if especially relevant and/or any distinctive features.
Keep it short and to the point.
'''
```