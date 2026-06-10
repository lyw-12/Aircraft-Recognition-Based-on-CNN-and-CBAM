import os
import cv2
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm
import shutil
import time
import random
import argparse

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='扩充测试集后的重复图片检测与处理工具')
    parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集根目录')
    parser.add_argument('--similarity_threshold', default=5, type=int, help='感知哈希差异阈值，越小越严格')
    parser.add_argument('--histogram_threshold', default=0.85, type=float, help='直方图相似度阈值')
    parser.add_argument('--mse_threshold', default=1000, type=int, help='均方误差阈值，越小表示越相似')
    parser.add_argument('--seed', default=42, type=int, help='随机种子')
    args = parser.parse_args()

    print("飞行器数据集优化工具 - 扩充测试集后的重复检测与处理")
    print("-" * 80)

    # 数据集目录
    dataset_dir = args.data_dir
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")
    duplicates_dir = os.path.join(dataset_dir, "duplicates_after_expansion")  # 存储检测到的重复图片

    # 确保存在用于保存重复图片的目录
    if not os.path.exists(duplicates_dir):
        os.makedirs(duplicates_dir)

    # 图像相似度比较的阈值
    SIMILARITY_THRESHOLD = args.similarity_threshold
    HISTOGRAM_THRESHOLD = args.histogram_threshold
    MSE_THRESHOLD = args.mse_threshold

    # 设置随机种子以确保结果可复现
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"使用参数: 感知哈希阈值={SIMILARITY_THRESHOLD}, 直方图阈值={HISTOGRAM_THRESHOLD}, MSE阈值={MSE_THRESHOLD}")
    print("开始检查训练集和测试集中的重复图片...")

    # 获取所有类别
    categories = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"发现类别: {categories}")

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

    print("\n收集训练集图片信息...")
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

    print("\n收集测试集图片信息...")
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

    print("\n开始检测重复图片...")
    for category in categories:
        print(f"检查类别 {category} 的重复图片...")

        # 为该类别创建存储重复图片的目录
        category_duplicates_dir = os.path.join(duplicates_dir, category)
        if not os.path.exists(category_duplicates_dir):
            os.makedirs(category_duplicates_dir)

        # 检查训练集与测试集之间的重复
        for train_idx, (train_img_path, train_hash, train_features) in enumerate(tqdm(train_images[category], desc=f"比较训练集 {category}")):
            for test_img_path, test_hash, test_features in test_images[category]:
                is_similar, similarity = are_images_similar(
                    train_img_path, test_img_path,
                    train_hash, test_hash,
                    train_features, test_features
                )

                if is_similar:
                    duplicates_found.append((train_img_path, test_img_path, similarity, category))

                    # 将重复图片复制到duplicates目录以便检查
                    train_img_name = os.path.basename(train_img_path)
                    test_img_name = os.path.basename(test_img_path)

                    duplicate_train_path = os.path.join(category_duplicates_dir, f"train_{train_img_name}")
                    duplicate_test_path = os.path.join(category_duplicates_dir, f"test_{test_img_name}")

                    shutil.copy(train_img_path, duplicate_train_path)
                    shutil.copy(test_img_path, duplicate_test_path)

    # 处理重复图片
    print(f"\n发现 {len(duplicates_found)} 对重复图片")

    if len(duplicates_found) > 0:
        # 创建备份目录
        backup_dir = os.path.join(dataset_dir, "train_backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # 按相似度排序（从高到低）
        duplicates_found.sort(key=lambda x: x[2], reverse=True)

        # 打印重复图片对
        print("\n重复图片详情:")
        for train_path, test_path, similarity, category in duplicates_found:
            print(f"类别: {category}, 相似度: {similarity:.4f}: {os.path.basename(train_path)} <=> {os.path.basename(test_path)}")

        # 统计每个类别的重复图片数量
        category_counts = {}
        for _, _, _, category in duplicates_found:
            if category not in category_counts:
                category_counts[category] = 0
            category_counts[category] += 1
        
        for category, count in category_counts.items():
            print(f"类别 {category}: {count} 对重复图片")

        # 询问用户是否要处理重复图片
        response = input("\n是否要从训练集中删除重复图片? (y/n): ")
        
        if response.lower() == 'y':
            print("\n处理重复图片...")
            removed_count = 0
            
            # 按类别创建备份目录
            for category in categories:
                category_backup_dir = os.path.join(backup_dir, category)
                if not os.path.exists(category_backup_dir):
                    os.makedirs(category_backup_dir)
            
            # 从训练集中移除重复图片
            for train_path, test_path, similarity, category in tqdm(duplicates_found, desc="移除重复图片"):
                train_img_name = os.path.basename(train_path)
                category_backup_dir = os.path.join(backup_dir, category)
                
                # 将训练集中的重复图片备份
                backup_path = os.path.join(category_backup_dir, train_img_name)
                
                try:
                    shutil.copy(train_path, backup_path)  # 先复制到备份目录
                    os.remove(train_path)  # 然后从训练集删除
                    removed_count += 1
                except Exception as e:
                    print(f"移动文件时出错: {str(e)}")
            
            print(f"\n处理完成! 已从训练集中移除 {removed_count} 张重复图片")
            
            # 打印最终统计
            print("\n数据集最终统计:")
            for category in categories:
                train_count = len([f for f in os.listdir(os.path.join(train_dir, category)) 
                                  if f.endswith(('.jpg', '.jpeg', '.png'))])
                test_count = len([f for f in os.listdir(os.path.join(test_dir, category)) 
                                 if f.endswith(('.jpg', '.jpeg', '.png'))])
                
                print(f"类别 {category}: 训练集 {train_count} 张，测试集 {test_count} 张")
        else:
            print("用户选择不处理重复图片。退出程序。")
    else:
        print("恭喜! 没有检测到训练集和测试集间的重复图片。")

    # 计算运行时间
    elapsed_time = time.time() - start_time
    print(f"\n程序运行时间: {elapsed_time:.2f} 秒")

if __name__ == '__main__':
    main() 