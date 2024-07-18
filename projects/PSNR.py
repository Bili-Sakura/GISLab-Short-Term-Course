import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io, color

def calculate_psnr(image_1, image_2, data_range=None):
    # 转换到YCbCr颜色空间
    image1_ycbcr = color.rgb2ycbcr(image_1)
    image2_ycbcr = color.rgb2ycbcr(image_2)
    
    # 提取Y通道
    y1 = image1_ycbcr[:, :, 0]/255.0
    y2 = image2_ycbcr[:, :, 0]/255.0

    return psnr(y1, y2, data_range=data_range)

def calculate_average_psnr(folder1, folder2):
    psnr_values = []

    images1 = {f: os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))}
    images2 = {f: os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))}
    
    # 找到两个文件夹中共有的文件名
    common_images = set(images1.keys()) & set(images2.keys())

    for image_name in common_images:
        image_1 = io.imread(os.path.join(folder1, image_name))
        image_2 = io.imread(os.path.join(folder2, image_name))

        # Calculate PSNR
        psnr_value = calculate_psnr(image_1, image_2, data_range=1.0)
        psnr_values.append(psnr_value)
        print(psnr_values)

    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        return average_psnr
    else:
        return None

# Replace with your folder paths
folder_path1 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/post'
folder_path2 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/pre'

# Calculate average PSNR
overall_psnr = calculate_average_psnr(folder_path1, folder_path2)
if overall_psnr is not None:
    print(f"The overall PSNR value for the image pairs in the two folders is: {overall_psnr} dB")
else:
    print("No valid image pairs found.")