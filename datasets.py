import os
import shutil
from torchvision import datasets
from torch.utils.data import random_split

# 数据集路径
dataset_dir = "/root/autodl-tmp/moe_sfda/datasets/brain_tumor_detection/figshare/dataset"
output_dir = "/root/autodl-tmp/moe_sfda/datasets/brain_tumor_detection/figshare"

# 创建输出文件夹
train_dir = os.path.join(output_dir, "Training")
test_dir = os.path.join(output_dir, "Testing")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 加载数据集
dataset = datasets.ImageFolder(root=dataset_dir)

# 数据集划分
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 创建类别子文件夹
def create_class_dirs(base_dir, classes):
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

# 创建类别文件夹
create_class_dirs(train_dir, dataset.classes)
create_class_dirs(test_dir, dataset.classes)

# 保存数据集到指定文件夹
def save_dataset(subset, target_dir):
    # 通过 Subset 的原始数据集获取类名
    original_dataset = subset.dataset  # 原始 ImageFolder 数据集
    for idx in subset.indices:  # 遍历 Subset 中的索引
        img_path, class_index = original_dataset.samples[idx]  # 获取图像路径和类别索引
        class_name = original_dataset.classes[class_index]  # 获取类别名称
        # 目标类别文件夹路径
        class_dir = os.path.join(target_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)  # 确保类别文件夹存在
        # 目标文件路径
        target_path = os.path.join(class_dir, os.path.basename(img_path))
        # 复制文件
        shutil.copy(img_path, target_path)

# 保存训练集和测试集
save_dataset(train_dataset, train_dir)
save_dataset(test_dataset, test_dir)

print(f"训练集保存到: {train_dir}")
print(f"测试集保存到: {test_dir}")
