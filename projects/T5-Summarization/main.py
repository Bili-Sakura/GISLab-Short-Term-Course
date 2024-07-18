import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Check if CUDA is available and set the device to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("./models/t5-large", legacy=False)
model = T5ForConditionalGeneration.from_pretrained("./models/t5-large").to(device)


# Function to summarize text
def summarize_text(text):
    input_ids = tokenizer.encode(
        "summarize: " + text, return_tensors="pt", max_length=512, truncation=True
    ).to(device)
    summary_ids = model.generate(
        input_ids,
        max_length=70,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Read from JSONL file and process each item
input_file = "./data/captions_demo.jsonl"
output_file = "./out/captions_summarization_demo.jsonl"

with open(input_file, "r") as reader, open(output_file, "w") as writer:
    for line in reader:
        obj = json.loads(line.strip())
        # Summarize the caption field
        caption = obj.get("caption", "")
        caption_short = summarize_text(caption)
        # Add the summarized caption to the object
        obj["caption_short"] = caption_short
        # Write the updated object to the output file
        writer.write(json.dumps(obj) + "\n")

print(f"Processed and saved summarized captions to {output_file}")
