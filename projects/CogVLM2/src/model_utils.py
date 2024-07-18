import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

def initialize_model(model_path, torch_type):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_type, trust_remote_code=True)
    return tokenizer, model

def get_device_map(model, max_memory_per_gpu, num_gpus):
    return infer_auto_device_map(
        model=model,
        max_memory={i: max_memory_per_gpu for i in range(num_gpus)},
        no_split_module_classes=["CogVLMDecoderLayer"]
    )

def load_model(model, model_path, device_map, torch_type):
    return load_checkpoint_and_dispatch(model, model_path, device_map=device_map, dtype=torch_type).eval()
