import os
import hashlib
import shutil
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger()

def compute_image_hash(img_path):
    """计算图像的哈希值"""
    try:
        # 打开图像并转换为numpy数组
        img = Image.open(img_path).convert('RGB')
        
        # 将图像调整为统一大小以确保哈希一致性
        img = img.resize((32, 32), Image.LANCZOS)
        img_array = np.array(img)
        
        # 将图像数组转换为字节并计算哈希
        return hashlib.md5(img_array.tobytes()).hexdigest(), img_path
    except Exception as e:
        logger.error(f"处理图像时出错 {img_path}: {str(e)}")
        return None, img_path

def get_all_images(directory):
    """获取目录中所有图像文件的路径"""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def find_duplicates(train_dir, val_dir, output_dir, max_workers=None):
    """查找训练集和验证集中的重复图像"""
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'duplicates'), exist_ok=True)
    
    # 获取所有图像路径
    logger.info("获取训练集图像路径...")
    train_images = get_all_images(train_dir)
    logger.info(f"训练集中找到 {len(train_images)} 张图像")
    
    logger.info("获取验证集图像路径...")
    val_images = get_all_images(val_dir)
    logger.info(f"验证集中找到 {len(val_images)} 张图像")
    
    # 计算所有图像的哈希值
    hash_results = {}
    total_images = len(train_images) + len(val_images)
    
    logger.info(f"开始计算 {total_images} 张图像的哈希值...")
    
    # 使用ProcessPoolExecutor进行并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有训练集图像的哈希计算任务
        train_futures = {executor.submit(compute_image_hash, img_path): ('train', img_path) for img_path in train_images}
        
        # 提交所有验证集图像的哈希计算任务
        val_futures = {executor.submit(compute_image_hash, img_path): ('val', img_path) for img_path in val_images}
        
        # 合并所有Future对象
        all_futures = {**train_futures, **val_futures}
        
        # 处理结果
        for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="计算图像哈希"):
            dataset, img_path = all_futures[future]
            try:
                img_hash, path = future.result()
                if img_hash:
                    if img_hash not in hash_results:
                        hash_results[img_hash] = []
                    hash_results[img_hash].append((dataset, path))
            except Exception as e:
                logger.error(f"处理 {img_path} 时出错: {str(e)}")
    
    # 查找重复图像
    logger.info("查找重复图像...")
    duplicates = {h: paths for h, paths in hash_results.items() if len(paths) > 1}
    
    # 创建结果数据框
    duplicate_data = []
    
    # 记录训练集和验证集之间的重复
    train_val_duplicates = 0
    
    # 处理重复图像
    logger.info(f"处理 {len(duplicates)} 组重复图像...")
    for img_hash, paths in duplicates.items():
        # 检查是否存在训练集和验证集之间的重复
        datasets = [p[0] for p in paths]
        is_cross_duplicate = 'train' in datasets and 'val' in datasets
        
        if is_cross_duplicate:
            train_val_duplicates += 1
        
        # 将每组重复的图像添加到数据框
        for i, (dataset, path) in enumerate(paths):
            # 复制重复图像到输出目录（仅限训练集和验证集之间的重复）
            if is_cross_duplicate:
                # 提取类别名称（假设路径格式为 .../dataset/class/image.jpg）
                class_name = os.path.basename(os.path.dirname(path))
                # 创建目标目录
                target_dir = os.path.join(output_dir, 'duplicates', f"group_{img_hash[:8]}")
                os.makedirs(target_dir, exist_ok=True)
                # 构建目标文件名（包含数据集信息）
                filename = f"{dataset}_{os.path.basename(path)}"
                # 复制文件
                shutil.copy2(path, os.path.join(target_dir, filename))
                
            # 添加到结果数据
            duplicate_data.append({
                'hash': img_hash,
                'group_id': img_hash[:8],
                'dataset': dataset,
                'class': os.path.basename(os.path.dirname(path)),
                'filename': os.path.basename(path),
                'path': path,
                'is_cross_duplicate': is_cross_duplicate
            })
    
    # 创建报告
    duplicate_df = pd.DataFrame(duplicate_data)
    
    # 保存完整的重复图像报告
    duplicate_df.to_csv(os.path.join(output_dir, 'duplicate_images_report.csv'), index=False)
    
    # 保存训练集和验证集之间的重复报告
    cross_duplicates_df = duplicate_df[duplicate_df['is_cross_duplicate'] == True]
    cross_duplicates_df.to_csv(os.path.join(output_dir, 'train_val_duplicates.csv'), index=False)
    
    # 按类别统计重复情况
    class_stats = duplicate_df[duplicate_df['is_cross_duplicate'] == True].groupby('class').size().reset_index(name='count')
    class_stats.to_csv(os.path.join(output_dir, 'duplicate_class_stats.csv'), index=False)
    
    # 生成摘要报告
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"重复图像检查摘要报告\n")
        f.write(f"======================\n\n")
        f.write(f"训练集图像总数: {len(train_images)}\n")
        f.write(f"验证集图像总数: {len(val_images)}\n")
        f.write(f"总图像数: {total_images}\n\n")
        f.write(f"重复图像组数: {len(duplicates)}\n")
        f.write(f"训练集和验证集之间的重复组数: {train_val_duplicates}\n\n")
        
        # 写入类别统计
        if not class_stats.empty:
            f.write(f"按类别的训练集-验证集重复统计:\n")
            for _, row in class_stats.iterrows():
                f.write(f"  {row['class']}: {row['count']} 张图像\n")
        
        f.write(f"\n处理时间: {time.time() - start_time:.2f} 秒\n")
    
    logger.info(f"检查完成! 结果保存在 {output_dir}")
    logger.info(f"训练集和验证集之间发现 {train_val_duplicates} 组重复图像")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查训练集和验证集中的重复图像')
    parser.add_argument('--train_dir', default='aircraft_dataset/train', type=str, help='训练集目录')
    parser.add_argument('--val_dir', default='aircraft_dataset/val', type=str, help='验证集目录')
    parser.add_argument('--output_dir', default='dataset-check', type=str, help='输出目录')
    parser.add_argument('--workers', type=int, default=None, help='并行处理的工作进程数')
    
    args = parser.parse_args()
    
    find_duplicates(args.train_dir, args.val_dir, args.output_dir, args.workers) 