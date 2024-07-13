import os
from PIL import Image, ImageDraw, ImageFont

def load_image(image_path):
    return Image.open(image_path)

def annotate_image(draw, text, position, font_size=16):
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, fill="black", font=font)

def combine_images(images, rows, cols):
    assert len(images) == rows * cols, "Number of images should match rows * cols"
    widths, heights = zip(*(img.size for img, _ in images))
    max_width = max(widths)
    max_height = max(heights)

    combined_width = max_width * cols
    combined_height = max_height * rows + rows * 20

    combined_image = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(combined_image)

    for idx, (img, text) in enumerate(images):
        row = idx // cols
        col = idx % cols
        x_offset = col * max_width
        y_offset = row * (max_height + 20)
        combined_image.paste(img, (x_offset, y_offset))
        annotate_image(draw, text, (x_offset + 10, y_offset + max_height + 5))

    return combined_image
