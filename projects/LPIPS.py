import os
import lpips
import torch
from PIL import Image
import numpy as np

# 初始化LPIPS模型
loss_fn = lpips.LPIPS(net='alex')

# 设置两个图像文件夹的路径
dir1 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/pre'  # 替换为第一个文件夹的路径
dir2 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/post'  # 替换为第二个文件夹的路径

# 获取两个文件夹中的文件列表
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)

# 找到两个文件夹中的同名文件
common_files = [file for file in files1 if file in files2]

# 存储LPIPS距离的列表
lpips_distances = []

# 遍历同名文件列表
for file in common_files:
    # 读取图像
    image1 = Image.open(os.path.join(dir1, file))
    image2 = Image.open(os.path.join(dir2, file))

    # 将图像转换为Tensor格式并归一化
    image1_tensor = torch.tensor(np.array(image1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    image2_tensor = torch.tensor(np.array(image2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # 计算LPIPS距离
    distance = loss_fn(image1_tensor, image2_tensor)
    lpips_distances.append(distance.item())

    # 打印当前图像的LPIPS距离
    print(f'{file}: LPIPS distance = {distance.item()}')

# 计算平均LPIPS距离
average_lpips = sum(lpips_distances) / len(lpips_distances)
print(f'Average LPIPS distance: {average_lpips}')