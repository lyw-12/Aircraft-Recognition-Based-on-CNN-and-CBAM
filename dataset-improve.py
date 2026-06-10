import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import shutil
import time
import random

print("飞行器数据集优化工具 - 训练集与测试集重复检测与处理")
print("-" * 80)

# 数据集目录
dataset_dir = r"C:\Users\32501\PycharmProjects\deeplearning\aircraft_dataset"
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
duplicates_dir = os.path.join(dataset_dir, "duplicates")  # 存储检测到的重复图片

# 确保存在用于保存重复图片的目录
if not os.path.exists(duplicates_dir):
    os.makedirs(duplicates_dir)

# 图像相似度比较的阈值（越小表示要求越严格）
SIMILARITY_THRESHOLD = 5  # 感知哈希差异阈值
HISTOGRAM_THRESHOLD = 0.85  # 直方图相似度阈值
MSE_THRESHOLD = 1000  # 均方误差阈值，值越小表示越相似

# 设置随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)

print("开始检查训练集和测试集中的重复图片...")

# 所有类别
categories = ["helicopter", "military_aircraft", "uav", "civil_airliner"]


def compute_image_hash(img_path):
    """计算图像的感知哈希值"""
    try:
        img = Image.open(img_path).convert('RGB')
        # 计算感知哈希
        phash = imagehash.phash(img)
        return phash
    except Exception as e:
        print(f"计算哈希值出错 {img_path}: {str(e)}")
        return None


def compute_image_features(img_path):
    """计算图像的特征向量用于相似度比较"""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        # 调整大小以加快计算
        img_resized = cv2.resize(img, (64, 64))
        # 计算图像的颜色直方图特征
        hist = cv2.calcHist([img_resized], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    except Exception as e:
        print(f"计算特征向量出错 {img_path}: {str(e)}")
        return None


def compute_mse_similarity(img1_path, img2_path):
    """使用均方误差(MSE)计算两张图片的相似度"""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return float('inf')  # 返回无穷大表示极不相似

        # 调整大小以匹配
        img1 = cv2.resize(img1, (140, 140))
        img2 = cv2.resize(img2, (140, 140))

        # 转为灰度图
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # 计算均方误差MSE
        err = np.sum((gray1.astype("float") - gray2.astype("float")) ** 2)
        err /= float(gray1.shape[0] * gray1.shape[1])

        return err
    except Exception as e:
        print(f"计算MSE相似度出错: {str(e)}")
        return float('inf')


def are_images_similar(img1_path, img2_path, img1_hash, img2_hash, img1_features, img2_features):
    """综合判断两张图片是否相似"""
    # 1. 首先用感知哈希快速筛选
    hash_diff = img1_hash - img2_hash
    if hash_diff > SIMILARITY_THRESHOLD:
        return False, 0

    # 2. 计算颜色直方图相似度
    hist_similarity = cv2.compareHist(img1_features, img2_features, cv2.HISTCMP_CORREL)
    if hist_similarity < HISTOGRAM_THRESHOLD:
        return False, 0

    # 3. 计算均方误差相似度(MSE)，值越小越相似
    mse = compute_mse_similarity(img1_path, img2_path)
    if mse > MSE_THRESHOLD:
        return False, 0

    # 综合相似度得分
    similarity_score = hist_similarity * (1 - mse / MSE_THRESHOLD / 2)

    return True, similarity_score


# 收集所有图片的路径和特征
train_images = {}  # {category: [(image_path, image_hash, image_features), ...]}
test_images = {}  # 同上

# 开始计时
start_time = time.time()

print("收集训练集图片信息...")
for category in categories:
    train_images[category] = []
    category_dir = os.path.join(train_dir, category)

    if not os.path.exists(category_dir):
        continue

    image_files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc=f"处理训练集 {category}"):
        img_path = os.path.join(category_dir, img_file)
        img_hash = compute_image_hash(img_path)
        img_features = compute_image_features(img_path)

        if img_hash is not None and img_features is not None:
            train_images[category].append((img_path, img_hash, img_features))

print("收集测试集图片信息...")
for category in categories:
    test_images[category] = []
    category_dir = os.path.join(test_dir, category)

    if not os.path.exists(category_dir):
        continue

    image_files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_files, desc=f"处理测试集 {category}"):
        img_path = os.path.join(category_dir, img_file)
        img_hash = compute_image_hash(img_path)
        img_features = compute_image_features(img_path)

        if img_hash is not None and img_features is not None:
            test_images[category].append((img_path, img_hash, img_features))

# 检测重复图片
duplicates_found = []  # 存储发现的重复图片对: [(train_img_path, test_img_path, similarity), ...]

print("开始检测重复图片...")
for category in categories:
    print(f"检查类别 {category} 的重复图片...")

    # 为该类别创建存储重复图片的目录
    category_duplicates_dir = os.path.join(duplicates_dir, category)
    if not os.path.exists(category_duplicates_dir):
        os.makedirs(category_duplicates_dir)

    # 检查训练集与测试集之间的重复
    for test_img_path, test_hash, test_features in tqdm(test_images[category], desc=f"比较 {category}"):
        for train_img_path, train_hash, train_features in train_images[category]:
            is_similar, similarity = are_images_similar(
                test_img_path, train_img_path,
                test_hash, train_hash,
                test_features, train_features
            )

            if is_similar:
                duplicates_found.append((train_img_path, test_img_path, similarity))

                # 将重复图片复制到duplicates目录
                train_img_name = os.path.basename(train_img_path)
                test_img_name = os.path.basename(test_img_path)

                duplicate_train_path = os.path.join(category_duplicates_dir, f"train_{train_img_name}")
                duplicate_test_path = os.path.join(category_duplicates_dir, f"test_{test_img_name}")

                shutil.copy(train_img_path, duplicate_train_path)
                shutil.copy(test_img_path, duplicate_test_path)

# 处理重复图片
print(f"发现 {len(duplicates_found)} 对重复图片")

if len(duplicates_found) > 0:
    # 创建替换图片目录
    replacements_dir = os.path.join(dataset_dir, "replacements")
    if not os.path.exists(replacements_dir):
        os.makedirs(replacements_dir)

    # 按相似度排序（从高到低）
    duplicates_found.sort(key=lambda x: x[2], reverse=True)

    # 打印重复图片对
    print("\n重复图片详情:")
    for train_path, test_path, similarity in duplicates_found:
        print(f"相似度 {similarity:.4f}: {os.path.basename(train_path)} <=> {os.path.basename(test_path)}")

    print("\n处理重复图片...")
    # 从训练集中移除重复图片，并从deleted_images或noisy_images中找替代图片
    for train_path, test_path, _ in duplicates_found:
        category = os.path.basename(os.path.dirname(train_path))
        train_img_name = os.path.basename(train_path)

        # 将训练集中的重复图片移动到replacements目录
        replacement_path = os.path.join(replacements_dir, f"{category}_{train_img_name}")
        shutil.move(train_path, replacement_path)
        print(f"已移动 {train_path} 到 {replacement_path}")

        # 尝试从deleted_images或noisy_images中找替代图片
        potential_sources = [
            os.path.join(dataset_dir, "deleted_images", category),
            os.path.join(dataset_dir, "noisy_images", category)
        ]

        replacement_found = False
        for source_dir in potential_sources:
            if os.path.exists(source_dir) and not replacement_found:
                available_images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

                if available_images:
                    # 随机选择一张替代图片
                    replacement_img = np.random.choice(available_images)
                    source_path = os.path.join(source_dir, replacement_img)

                    # 确保替代图片与测试集中的所有图片都不重复
                    is_duplicate = False
                    replace_hash = compute_image_hash(source_path)
                    replace_features = compute_image_features(source_path)

                    if replace_hash is not None and replace_features is not None:
                        for test_img_path, test_hash, test_features in test_images[category]:
                            is_similar, _ = are_images_similar(
                                source_path, test_img_path,
                                replace_hash, test_hash,
                                replace_features, test_features
                            )
                            if is_similar:
                                is_duplicate = True
                                break

                    if not is_duplicate:
                        # 复制到训练集中
                        shutil.copy(source_path, train_path)
                        print(f"已用 {source_path} 替换 {train_path}")
                        replacement_found = True

        if not replacement_found:
            print(f"警告: 无法为 {train_path} 找到合适的替代图片")

print("\n检测和处理重复图片完成")

# 统计更新后的数据集信息
train_counts = {}
test_counts = {}

for category in categories:
    train_path = os.path.join(train_dir, category)
    test_path = os.path.join(test_dir, category)

    if os.path.exists(train_path):
        train_count = len([f for f in os.listdir(train_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        train_counts[category] = train_count

    if os.path.exists(test_path):
        test_count = len([f for f in os.listdir(test_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        test_counts[category] = test_count

print("\n更新后的数据集统计:")
print("训练集:")
for category, count in train_counts.items():
    print(f"  - {category}: {count} 张图片")

print("测试集:")
for category, count in test_counts.items():
    print(f"  - {category}: {count} 张图片")

total_train = sum(train_counts.values())
total_test = sum(test_counts.values())
total_images = total_train + total_test

print(f"训练集总计: {total_train} 张图片")
print(f"测试集总计: {total_test} 张图片")
print(f"总计: {total_images} 张图片")

# 显示处理时间
end_time = time.time()
process_time = end_time - start_time
print(f"\n总处理时间: {process_time:.2f} 秒")
print(f"平均每张图片处理时间: {process_time / total_images * 1000:.2f} 毫秒")

print("\n数据集优化完成!")
print("-" * 80)