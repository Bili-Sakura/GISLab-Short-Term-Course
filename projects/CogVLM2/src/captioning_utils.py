import torch
from PIL import Image

def generate_caption(model, tokenizer, image, prompt, device, torch_type):
    input_by_model = model.build_conversation_input_ids(
        tokenizer,
        query=prompt,
        history=[],
        images=[image],
        template_version="chat"
    )

    inputs = {
        "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
        "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
        "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
        "images": [[input_by_model["images"][0].to(device).to(torch_type)]],
    }

    gen_kwargs = {
        "max_new_tokens": 2048,
        "pad_token_id": tokenizer.pad_token_id,
        "top_k": 1,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        caption = response.split("ASSISTANT: ")[-1].strip()
    
    return caption
