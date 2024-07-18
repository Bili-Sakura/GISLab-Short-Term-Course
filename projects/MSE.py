import os
import numpy as np
from PIL import Image

def mse(image1_path, image2_path):
    """计算两张图像的均方误差"""
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # 将图像转换为NumPy数组
    image1_array = np.array(image1)
    image2_array = np.array(image2)
    
    # 计算MSE
    mse_value = np.mean((image1_array - image2_array) ** 2)
    return mse_value

# 设置两个图像文件夹的路径
dir1 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/pre'  # 替换为第一个文件夹的路径
dir2 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/post'  # 替换为第二个文件夹的路径

# 获取两个文件夹中的文件列表
files1 = os.listdir(dir1)
files2 = os.listdir(dir2)

# 找到两个文件夹中的同名文件
common_files = [file for file in files1 if file in files2]

# 存储MSE距离的列表
mse_distances = []

# 遍历同名文件列表
for file in common_files:
    # 读取图像路径
    image1_path = os.path.join(dir1, file)
    image2_path = os.path.join(dir2, file)
    
    # 计算MSE
    mse_value = mse(image1_path, image2_path)
    mse_distances.append(mse_value)
    
    # 打印当前图像的MSE
    print(f'{file}: MSE = {mse_value}')

# 计算平均MSE
average_mse = sum(mse_distances) / len(mse_distances)
print(f'Average MSE: {average_mse}')