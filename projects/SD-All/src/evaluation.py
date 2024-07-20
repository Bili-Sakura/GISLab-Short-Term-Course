# src/evaluation.py

import os
import numpy as np
import torch
from PIL import Image
from skimage import io, color
from skimage.metrics import (
    peak_signal_noise_ratio as psnr,
    structural_similarity as ssim,
)
import lpips


# LPIPS Evaluation
def calculate_lpips(dir1, dir2):
    loss_fn = lpips.LPIPS(net="alex")
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    common_files = [file for file in files1 if file in files2]

    lpips_distances = []
    for file in common_files:
        image1 = Image.open(os.path.join(dir1, file))
        image2 = Image.open(os.path.join(dir2, file))
        image1_tensor = (
            torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        image2_tensor = (
            torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )
        distance = loss_fn(image1_tensor, image2_tensor)
        lpips_distances.append(distance.item())
        print(f"{file}: LPIPS distance = {distance.item()}")

    average_lpips = sum(lpips_distances) / len(lpips_distances)
    print(f"Average LPIPS distance: {average_lpips}")


# MSE Evaluation
def mse(image1_path, image2_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    mse_value = np.mean((image1_array - image2_array) ** 2)
    return mse_value


def calculate_average_mse(dir1, dir2):
    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    common_files = [file for file in files1 if file in files2]

    mse_distances = []
    for file in common_files:
        image1_path = os.path.join(dir1, file)
        image2_path = os.path.join(dir2, file)
        mse_value = mse(image1_path, image2_path)
        mse_distances.append(mse_value)
        print(f"{file}: MSE = {mse_value}")

    average_mse = sum(mse_distances) / len(mse_distances)
    print(f"Average MSE: {average_mse}")


# PSNR Evaluation
def calculate_psnr(image_1, image_2, data_range=None):
    image1_ycbcr = color.rgb2ycbcr(image_1)
    image2_ycbcr = color.rgb2ycbcr(image_2)
    y1 = image1_ycbcr[:, :, 0] / 255.0
    y2 = image2_ycbcr[:, :, 0] / 255.0
    return psnr(y1, y2, data_range=data_range)


def calculate_average_psnr(folder1, folder2):
    psnr_values = []
    images1 = {
        f: os.path.join(folder1, f)
        for f in os.listdir(folder1)
        if f.endswith((".png", ".jpg", ".jpeg"))
    }
    images2 = {
        f: os.path.join(folder2, f)
        for f in os.listdir(folder2)
        if f.endswith((".png", ".jpg", ".jpeg"))
    }
    common_images = set(images1.keys()) & set(images2.keys())

    for image_name in common_images:
        image_1 = io.imread(os.path.join(folder1, image_name))
        image_2 = io.imread(os.path.join(folder2, image_name))
        psnr_value = calculate_psnr(image_1, image_2, data_range=1.0)
        psnr_values.append(psnr_value)
        print(psnr_values)

    if psnr_values:
        average_psnr = sum(psnr_values) / len(psnr_values)
        print(f"Average PSNR: {average_psnr} dB")
    else:
        print("No valid image pairs found.")


# SSIM Evaluation
def calculate_ssim_for_pair(image_path1, image_path2):
    image1 = io.imread(image_path1)
    image2 = io.imread(image_path2)
    image1_ycbcr = color.rgb2ycbcr(image1)
    image2_ycbcr = color.rgb2ycbcr(image2)
    y1 = image1_ycbcr[:, :, 0] / 255.0
    y2 = image2_ycbcr[:, :, 0] / 255.0
    return ssim(y1, y2, data_range=1, multichannel=False)


def calculate_average_ssim(folder1, folder2):
    images1 = {
        f: os.path.join(folder1, f)
        for f in os.listdir(folder1)
        if f.endswith((".png", ".jpg", ".jpeg"))
    }
    images2 = {
        f: os.path.join(folder2, f)
        for f in os.listdir(folder2)
        if f.endswith((".png", ".jpg", ".jpeg"))
    }
    common_images = set(images1.keys()) & set(images2.keys())

    ssim_values = []
    for image_name in common_images:
        ssim_value = calculate_ssim_for_pair(images1[image_name], images2[image_name])
        ssim_values.append(ssim_value)
    print(ssim_values)

    average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    print(f"Average SSIM: {average_ssim}")


if __name__ == "__main__":
    dir1 = "E:/2024duanxueqi/Dataset/Detailed_Dataset_100/pre"
    dir2 = "E:/2024duanxueqi/Dataset/Detailed_Dataset_100/post"

    calculate_lpips(dir1, dir2)
    calculate_average_mse(dir1, dir2)
    calculate_average_psnr(dir1, dir2)
    calculate_average_ssim(dir1, dir2)
