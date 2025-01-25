import os
import shutil
import pandas as pd

def organize_images_by_category(image_dir, csv_path, output_dir):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 遍历CSV文件中的每一行
    for _, row in df.iterrows():
        image_name = row[0]  # 第一列是图片文件名
        category = row[1]    # 第二列是类别

        # 构建类别文件夹路径
        category_dir = os.path.join(output_dir, str(category))

        # 如果类别文件夹不存在，创建它
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)

        # 构建源文件和目标文件路径
        source_path = os.path.join(image_dir, image_name)
        target_path = os.path.join(category_dir, image_name)

        # 如果源文件存在，将其移动到目标文件夹
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
        else:
            print(f"Warning: {image_name} does not exist in {image_dir}")

if __name__ == "__main__":
    # 指定图片文件夹路径、CSV文件路径和输出文件夹路径
    image_dir = "/home/zjc/project/dataset/plant-pathology-2021-fgvc8/train_images"  # 替换为你的图片文件夹路径
    csv_path = "/home/zjc/project/dataset/plant-pathology-2021-fgvc8/train.csv"  # 替换为你的CSV文件路径
    output_dir = "/home/zjc/project/dataset/plant-pathology"  # 替换为输出文件夹路径

    organize_images_by_category(image_dir, csv_path, output_dir)
    print("Images have been organized by category.")
