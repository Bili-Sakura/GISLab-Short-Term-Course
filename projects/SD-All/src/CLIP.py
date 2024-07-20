import torch
import clip
from PIL import Image

# 加载CLIP模型和预训练的权重
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def compute_clip_score(image_path, text):
    # 加载图像并预处理
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    
    # 编码图像和文本
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(clip.tokenize([text]).to(device))
    
    # 计算余弦相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (image_features @ text_features.T).item()
    
    return similarity

# 示例使用
image_path = "path/to/image"
text_description = "A description of the image"
clip_score = compute_clip_score(image_path, text_description)
print(f"CLIP-Score: {clip_score}")

