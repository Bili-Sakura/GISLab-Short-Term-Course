import os
from skimage import io, color
from skimage.metrics import structural_similarity as ssim

def calculate_ssim_for_pair(image_path1, image_path2):
    # 读取图像
    image1 = io.imread(image_path1)
    image2 = io.imread(image_path2)
    
    # 转换到YCbCr颜色空间
    image1_ycbcr = color.rgb2ycbcr(image1)
    image2_ycbcr = color.rgb2ycbcr(image2)
    
    # 提取Y通道
    y1 = image1_ycbcr[:, :, 0]/255.0
    y2 = image2_ycbcr[:, :, 0]/255.0
    # 计算Y通道的SSIM
    return ssim(y1, y2, data_range=1, multichannel=False)

def calculate_average_ssim(folder1, folder2):
    # 获取两个文件夹中的所有图像文件路径
    images1 = {f: os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('.png', '.jpg', '.jpeg'))}
    images2 = {f: os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('.png', '.jpg', '.jpeg'))}
    
    # 找到两个文件夹中共有的文件名
    common_images = set(images1.keys()) & set(images2.keys())
    
    # 计算所有共有图像对的SSIM值
    ssim_values = []
    for image_name in common_images:
        ssim_value = calculate_ssim_for_pair(images1[image_name], images2[image_name])
        ssim_values.append(ssim_value)
    print(ssim_values)    
    
    # 计算平均SSIM值
    average_ssim = sum(ssim_values) / len(ssim_values) if ssim_values else 0
    return average_ssim

# 替换为你的两个图像文件夹路径
folder_path1 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/post'
folder_path2 = 'E:/2024duanxueqi/Dataset/Detailed_Dataset_100/pre'

# 计算平均SSIM
overall_ssim = calculate_average_ssim(folder_path1, folder_path2)
print(f"The overall SSIM value for the common images in the two folders is: {overall_ssim}")