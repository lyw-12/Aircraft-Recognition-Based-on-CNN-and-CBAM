import os
import random
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter
import time
import shutil

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"当前目录: {current_dir}")

# 明确指定数据集路径
dataset_dir = r"C:\Users\32501\PycharmProjects\deeplearning\aircraft_dataset"
if not os.path.exists(dataset_dir):
    # 尝试一些可能的位置
    possible_locations = [
        r"C:\Users\32501\PycharmProjects\deeplearning\aircraft_dataset",
        os.path.join(current_dir, "aircraft_dataset"),
        os.path.abspath(os.path.join(current_dir, os.pardir, "aircraft_dataset")),
        os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, "aircraft_dataset"))
    ]

    for location in possible_locations:
        print(f"尝试查找数据集: {location}")
        if os.path.exists(location):
            dataset_dir = location
            print(f"找到数据集目录: {dataset_dir}")
            break

    if not os.path.exists(dataset_dir):
        # 如果找不到，尝试从当前目录向上找
        print("未找到预设数据集目录，尝试查找...")
        current_path = current_dir
        for _ in range(4):  # 向上最多查找4级
            parent_dir = os.path.dirname(current_path)
            test_path = os.path.join(parent_dir, "aircraft_dataset")
            print(f"尝试: {test_path}")
            if os.path.exists(test_path):
                dataset_dir = test_path
                print(f"找到数据集目录: {dataset_dir}")
                break
            current_path = parent_dir

print(f"最终使用的数据集目录: {dataset_dir}")
if not os.path.exists(dataset_dir):
    raise FileNotFoundError(f"找不到数据集目录！请确保aircraft_dataset目录存在，或修改脚本中的dataset_dir变量为正确路径。")

train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
print(f"训练集目录: {train_dir}")
print(f"测试集目录: {test_dir}")

# 检查训练集目录是否存在
if not os.path.exists(train_dir):
    raise FileNotFoundError(f"训练集目录不存在: {train_dir}")

# 创建临时目录用于存放被删除的图片（以防万一需要恢复）
deleted_dir = os.path.join(dataset_dir, "deleted_images")
os.makedirs(deleted_dir, exist_ok=True)

# 清理后每个类别目标数量
target_count = 5000
categories = ["helicopter", "military_aircraft", "uav", "civil_airliner"]

# 创建临时文件夹存储各类别删除的图像
for category in categories:
    os.makedirs(os.path.join(deleted_dir, category), exist_ok=True)


# 图像质量评估函数
def assess_image_quality(img_path):
    """评估图像质量并返回分数"""
    try:
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            return 0

        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算拉普拉斯方差（清晰度度量）
        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 计算亮度和对比度
        brightness = np.mean(gray)
        contrast = np.std(gray)

        # 图像尺寸
        height, width = img.shape[:2]
        size_score = min(width, height) / 140  # 与目标尺寸的比例

        # 检测边缘 - 通常飞行器具有明显的边缘
        edges = cv2.Canny(gray, 100, 200)
        edge_score = np.count_nonzero(edges) / (width * height)

        # 综合分数
        score = (clarity * 0.3 +
                 contrast * 0.2 +
                 min(brightness, 255 - brightness) * 0.1 +
                 size_score * 0.2 +
                 edge_score * 100 * 0.2)  # 边缘得分通常很小，所以要乘以更大的权重

        return score
    except Exception as e:
        print(f"评估图像质量时出错: {str(e)}")
        return 0


# 高质量数据增强函数
def augment_image(img, filename=None):
    """对图像进行数据增强，返回一个增强后的新图像"""
    # 基本增强技术列表
    basic_augmentations = [
        lambda x: x.rotate(random.randint(-15, 15), expand=False),  # 小角度旋转
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),  # 水平翻转
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),  # 垂直翻转
    ]

    # 更多高级变换
    advanced_augmentations = [
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.85, 1.15)),  # 亮度调整
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.85, 1.15)),  # 对比度调整
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.9, 1.1)),  # 颜色调整
    ]

    # 首先应用一种基本变换（必须）
    if filename and "flip" in filename.lower():
        # 如果文件名已经包含flip，避免重复翻转
        aug_img = img.copy()
    else:
        aug_func = random.choice(basic_augmentations)
        aug_img = aug_func(img)

    # 随机决定是否应用高级变换
    if random.random() < 0.5:
        adv_func = random.choice(advanced_augmentations)
        aug_img = adv_func(aug_img)

    # 随机裁剪，确保裁剪区域包含中心
    width, height = aug_img.size
    min_dim = min(width, height)

    # 在原始尺寸的80%-95%之间随机选择裁剪尺寸
    crop_size = int(min_dim * random.uniform(0.8, 0.95))

    # 计算裁剪区域，确保包含中心
    left = (width - crop_size) // 2 + random.randint(-width // 10, width // 10)
    top = (height - crop_size) // 2 + random.randint(-height // 10, height // 10)

    # 修正坐标以确保在图像范围内
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))

    right = left + crop_size
    bottom = top + crop_size

    # 裁剪并调整为140x140
    cropped_img = aug_img.crop((left, top, right, bottom))
    resized_img = cropped_img.resize((140, 140), Image.LANCZOS)

    return resized_img


# 清理低质量图像并生成高质量增强图像
def clean_and_augment():
    # 处理训练集数据
    for category in categories:
        train_category_dir = os.path.join(train_dir, category)

        if not os.path.exists(train_category_dir):
            print(f"警告: 类别目录不存在: {train_category_dir}")
            continue

        # 获取该类别下的所有图像
        print(f"\n开始处理类别: {category}")
        image_files = []

        for img_file in os.listdir(train_category_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.jpeg') or img_file.endswith('.png'):
                img_path = os.path.join(train_category_dir, img_file)
                quality_score = assess_image_quality(img_path)
                image_files.append((img_file, img_path, quality_score))

        # 按质量评分排序
        image_files.sort(key=lambda x: x[2], reverse=True)

        # 显示当前数量
        current_count = len(image_files)
        print(f"类别 {category}: 当前有 {current_count} 张图片")

        # 如果图片数量过多，则删除低质量图片
        if current_count > target_count:
            # 保留高质量图片
            keep_images = image_files[:target_count]
            # 低质量图片
            low_quality_images = image_files[target_count:]

            print(f"删除 {len(low_quality_images)} 张低质量图片")
            for img_file, img_path, _ in tqdm(low_quality_images, desc=f"删除低质量 {category} 图片"):
                # 移动到删除文件夹而不是直接删除
                dest_path = os.path.join(deleted_dir, category, img_file)
                shutil.move(img_path, dest_path)

            # 更新当前数量
            current_count = len(keep_images)
        else:
            keep_images = image_files

        # 如果图片数量不足，生成高质量增强图片
        if current_count < target_count:
            # 需要生成的数量
            need_to_generate = target_count - current_count
            print(f"需要生成 {need_to_generate} 张增强图片")

            # 重新获取剩余的高质量图片
            high_quality_imgs = [(img_file, img_path) for img_file, img_path, _ in keep_images]

            # 生成高质量增强图片
            generated_count = 0

            # 使用tqdm显示进度
            with tqdm(total=need_to_generate, desc=f"生成 {category} 增强图片") as pbar:
                while generated_count < need_to_generate:
                    # 随机选择一张高质量图片
                    img_file, img_path = random.choice(high_quality_imgs)

                    try:
                        # 打开图片
                        with Image.open(img_path) as img:
                            img = img.convert('RGB')

                            # 增强图片
                            aug_img = augment_image(img, img_file)

                            # 生成新文件名
                            base_index = current_count + generated_count
                            if "aug" in img_file:
                                new_img_name = f"{category}_{base_index:05d}_aug2.jpg"
                            else:
                                new_img_name = f"{category}_{base_index:05d}_aug.jpg"

                            new_img_path = os.path.join(train_category_dir, new_img_name)

                            # 保存增强后的图片
                            aug_img.save(new_img_path, "JPEG", quality=95)

                            generated_count += 1
                            pbar.update(1)

                            # 每处理10张图片暂停一下，避免系统过载
                            if generated_count % 10 == 0:
                                time.sleep(0.01)

                    except Exception as e:
                        print(f"处理图片 {img_file} 时出错: {str(e)}")

    # 处理完成后统计数据
    print("\n数据增强和清理完成！")
    print("更新后的训练集统计：")
    for category in categories:
        category_path = os.path.join(train_dir, category)
        if os.path.isdir(category_path):
            count = len([f for f in os.listdir(category_path)
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  - {category}: {count} 张图片")

    # 计算总数
    total_train = sum(len([f for f in os.listdir(os.path.join(train_dir, cat))
                           if f.endswith(('.jpg', '.jpeg', '.png'))])
                      for cat in categories if os.path.isdir(os.path.join(train_dir, cat)))

    total_test = 0
    if os.path.exists(test_dir):
        total_test = sum(len([f for f in os.listdir(os.path.join(test_dir, cat))
                              if f.endswith(('.jpg', '.jpeg', '.png'))])
                         for cat in categories if os.path.isdir(os.path.join(test_dir, cat)))

    print(f"总计: {total_train + total_test} 张图片 (训练: {total_train}, 测试: {total_test})")


if __name__ == "__main__":
    clean_and_augment()