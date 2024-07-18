# test/cli_demo_multi_gpus.py
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
)

MODEL_PATH = "./models/cogvlm2-llama3-chat-19B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Initialize the model with empty weights
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True
    )

# Determine the number of available GPUs and log the information
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for i in range(num_gpus):
    print(
        f"GPU {i} - Memory: {torch.cuda.get_device_properties(i).total_memory / (1024**3)} GiB"
    )

# Infer the device map and load the model across GPUs
max_memory_per_gpu = "23GiB"  # Use 23GiB per GPU to provide some buffer
device_map = infer_auto_device_map(
    model=model,
    max_memory={i: max_memory_per_gpu for i in range(3)},
    # max_memory={0: max_memory_per_gpu, 1: max_memory_per_gpu, 3: max_memory_per_gpu},
    no_split_module_classes=["CogVLMDecoderLayer"],
)
print("Device map:", device_map)

# Load model with checkpoint and dispatch across GPUs
model = load_checkpoint_and_dispatch(
    model, MODEL_PATH, device_map=device_map, dtype=TORCH_TYPE
)
model = model.eval()

# Template for text-only conversation
text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

# Interactive chat loop
while True:
    image_path = input("image path >>>>> ")
    if image_path == "":
        print(
            "You did not enter an image path, the following will be a plain text conversation."
        )
        image = None
        text_only_first_query = True
    else:
        try:

            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print("Image not found. Continuing with text-only conversation.")
            image = None
            text_only_first_query = True

    history = []

    while True:
        query = input("Human: ")
        if query.lower() == "clear":
            break

        if image is None:
            if text_only_first_query:
                query = text_only_template.format(query)
                text_only_first_query = False
            else:
                old_prompt = "".join(
                    [f"USER: {q} ASSISTANT: {r}\n" for q, r in history]
                )
                query = old_prompt + f"USER: {query} ASSISTANT:"

        if image is None:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, template_version="chat"
            )
        else:
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version="chat",
            )

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
            "images": (
                [[input_by_model["images"][0].to(DEVICE).to(TORCH_TYPE)]]
                if image is not None
                else None
            ),
        }

        gen_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": tokenizer.pad_token_id,
            "top_k": 1,
        }

        with torch.no_grad():
            try:
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response.split("ASSISTANT: ")[-1].strip()
                print("\nCogVLM2:", response)
            except torch.cuda.OutOfMemoryError:
                print(
                    "CUDA out of memory. Please reduce the input size or restart the program."
                )

        history.append((query, response))
