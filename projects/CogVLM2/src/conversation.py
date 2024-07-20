import torch
from PIL import Image

text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

def interactive_chat(model, tokenizer, device, torch_type):
    while True:
        image_path = input("image path >>>>> ")
        if image_path == "":
            print("You did not enter an image path, the following will be a plain text conversation.")
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
                    old_prompt = "".join([f"USER: {q} ASSISTANT: {r}\n" for q, r in history])
                    query = old_prompt + f"USER: {query} ASSISTANT:"

            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image] if image is not None else None,
                template_version="chat"
            )

            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to(device),
                "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(device),
                "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(device),
                "images": ([[input_by_model["images"][0].to(device).to(torch_type)]] if image is not None else None),
            }

            gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": tokenizer.pad_token_id,
                "top_k": 1,
            }

            with torch.no_grad():
                try:
                    outputs = model.generate(**inputs, **gen_kwargs)
                    outputs = outputs[:, inputs["input_ids"].shape[1]:]
                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = response.split("ASSISTANT: ")[-1].strip()
                    print("\nCogVLM2:", response)
                except torch.cuda.OutOfMemoryError:
                    print("CUDA out of memory. Please reduce the input size or restart the program.")

            history.append((query, response))
