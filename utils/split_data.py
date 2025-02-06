import os
import shutil
import random

# 原始数据集路径
dataset_dir = "/home/zjc/project/dataset/riceLeaf_raw"
output_dir = "/home/zjc/project/dataset/riceLeaf"

# 设置划分比例
train_ratio = 0.8
test_ratio = 0.2

# 确保输出文件夹存在
for split in ['train', 'test']:
    for class_name in os.listdir(dataset_dir):
        os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

# 开始划分
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    images = os.listdir(class_dir)
    random.shuffle(images)  # 随机打乱

    train_count = int(len(images) * train_ratio)
    test_count = int(len(images) * test_ratio)

    train_images = images[:train_count]
    test_images = images[train_count:train_count + test_count]

    for img in train_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'train', class_name, img))
    for img in test_images:
        shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'test', class_name, img))

print("数据集划分完成！")
