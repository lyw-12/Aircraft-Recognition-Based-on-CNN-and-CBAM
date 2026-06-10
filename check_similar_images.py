import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import time
import logging
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
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

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, image_path
        except Exception as e:
            logger.error(f"处理图像 {self.image_paths[idx]} 时出错: {str(e)}")
            # 返回一个空图像
            dummy_image = torch.zeros((3, 224, 224))
            return dummy_image, self.image_paths[idx]

def get_all_images(directory):
    """获取目录中所有图像文件的路径"""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_features(model, dataloader):
    """提取图像特征向量"""
    features = []
    paths = []
    
    model.eval()
    with torch.no_grad():
        for images, image_paths in tqdm(dataloader, desc="提取特征"):
            images = images.to(device)
            
            # 前向传播获取特征
            feat = model(images)
            
            # 将特征添加到列表
            features.append(feat.cpu().numpy())
            paths.extend(image_paths)
    
    # 合并所有特征
    features = np.vstack(features)
    
    return features, paths

def find_similar_images(train_dir, val_dir, output_dir, batch_size=32, similarity_threshold=0.95):
    """查找训练集和验证集中的相似图像"""
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'similar_pairs'), exist_ok=True)
    
    # 获取所有图像路径
    logger.info("获取训练集和验证集图像路径...")
    train_images = get_all_images(train_dir)
    val_images = get_all_images(val_dir)
    
    logger.info(f"训练集中找到 {len(train_images)} 张图像")
    logger.info(f"验证集中找到 {len(val_images)} 张图像")
    
    # 加载预训练模型
    logger.info("加载预训练的ResNet模型作为特征提取器...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 移除最后的全连接层，使用特征提取器
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model = model.to(device)
    
    # 创建数据加载器
    train_dataset = ImageDataset(train_images, preprocess)
    val_dataset = ImageDataset(val_images, preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 提取特征
    logger.info("从训练集图像中提取特征...")
    train_features, train_paths = extract_features(model, train_loader)
    
    logger.info("从验证集图像中提取特征...")
    val_features, val_paths = extract_features(model, val_loader)
    
    # 计算训练集和验证集之间的相似度
    logger.info("计算图像之间的相似度...")
    
    # 预处理特征向量，将每个向量归一化
    train_features = train_features.reshape(train_features.shape[0], -1)
    val_features = val_features.reshape(val_features.shape[0], -1)
    
    # L2归一化
    train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
    val_features = val_features / np.linalg.norm(val_features, axis=1, keepdims=True)
    
    # 计算余弦相似度
    logger.info("计算训练集和验证集之间的余弦相似度...")
    
    # 由于可能的内存限制，分批计算相似度
    batch_size = 1000
    similar_pairs = []
    
    for i in range(0, len(train_features), batch_size):
        train_batch = train_features[i:i+batch_size]
        
        # 计算当前批次与所有验证集图像的相似度
        similarities = cosine_similarity(train_batch, val_features)
        
        # 找出相似度高于阈值的对
        for j, similarity_row in enumerate(similarities):
            train_idx = i + j
            
            # 找到所有相似度高于阈值的验证集图像
            similar_val_indices = np.where(similarity_row >= similarity_threshold)[0]
            
            for val_idx in similar_val_indices:
                similar_pairs.append({
                    'train_path': train_paths[train_idx],
                    'val_path': val_paths[val_idx],
                    'similarity': similarity_row[val_idx],
                    'train_class': os.path.basename(os.path.dirname(train_paths[train_idx])),
                    'val_class': os.path.basename(os.path.dirname(val_paths[val_idx]))
                })
                
        logger.info(f"处理了 {i+len(train_batch)}/{len(train_features)} 个训练集图像")
    
    # 转换为DataFrame并按相似度降序排序
    similar_df = pd.DataFrame(similar_pairs)
    similar_df = similar_df.sort_values('similarity', ascending=False).reset_index(drop=True)
    
    # 保存相似图像对列表
    similar_df.to_csv(os.path.join(output_dir, 'similar_images.csv'), index=False)
    
    # 按类别统计相似图像对
    class_counts = similar_df.groupby(['train_class', 'val_class']).size().reset_index(name='count')
    class_counts.to_csv(os.path.join(output_dir, 'similar_images_by_class.csv'), index=False)
    
    # 复制相似图像对到输出目录
    logger.info("保存相似图像对的样例...")
    
    # 只保存每个类别前10对相似图像
    class_pairs = similar_df.groupby(['train_class', 'val_class'])
    for (train_class, val_class), group in class_pairs:
        # 取排序后的前10对
        top_pairs = group.head(10)
        
        for idx, row in top_pairs.iterrows():
            # 创建保存目录
            pair_dir = os.path.join(output_dir, 'similar_pairs', f"{train_class}_{val_class}")
            os.makedirs(pair_dir, exist_ok=True)
            
            # 构建文件名 (使用相似度作为前缀)
            similarity_str = f"{row['similarity']:.4f}"
            train_filename = f"train_{similarity_str}_{os.path.basename(row['train_path'])}"
            val_filename = f"val_{similarity_str}_{os.path.basename(row['val_path'])}"
            
            # 复制文件
            shutil.copy2(row['train_path'], os.path.join(pair_dir, train_filename))
            shutil.copy2(row['val_path'], os.path.join(pair_dir, val_filename))
            
            # 创建并保存图像对比图
            compare_path = os.path.join(pair_dir, f"compare_{similarity_str}.jpg")
            create_comparison_image(row['train_path'], row['val_path'], compare_path, row['similarity'])
    
    # 生成摘要报告
    total_similar_pairs = len(similar_df)
    with open(os.path.join(output_dir, 'similar_images_summary.txt'), 'w') as f:
        f.write(f"相似图像检查摘要报告\n")
        f.write(f"======================\n\n")
        f.write(f"训练集图像总数: {len(train_images)}\n")
        f.write(f"验证集图像总数: {len(val_images)}\n")
        f.write(f"相似度阈值: {similarity_threshold}\n")
        f.write(f"发现的相似图像对数量: {total_similar_pairs}\n\n")
        
        # 按类别统计
        f.write(f"按类别的相似图像对统计:\n")
        for _, row in class_counts.iterrows():
            f.write(f"  {row['train_class']} (训练) - {row['val_class']} (验证): {row['count']} 对\n")
        
        # 总结信息
        same_class_pairs = similar_df[similar_df['train_class'] == similar_df['val_class']]
        diff_class_pairs = similar_df[similar_df['train_class'] != similar_df['val_class']]
        
        f.write(f"\n同类相似图像对: {len(same_class_pairs)} ({len(same_class_pairs)/total_similar_pairs*100:.2f}%)\n")
        f.write(f"不同类相似图像对: {len(diff_class_pairs)} ({len(diff_class_pairs)/total_similar_pairs*100:.2f}%)\n")
        f.write(f"\n处理时间: {time.time() - start_time:.2f} 秒\n")
    
    logger.info(f"检查完成! 结果保存在 {output_dir}")
    logger.info(f"发现 {total_similar_pairs} 对相似图像")

def create_comparison_image(img1_path, img2_path, output_path, similarity):
    """创建两张图像的对比图"""
    # 打开两张图像
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')
    
    # 调整大小
    size = (224, 224)
    img1 = img1.resize(size)
    img2 = img2.resize(size)
    
    # 创建matplotlib图表
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 显示图像
    axes[0].imshow(np.array(img1))
    axes[0].set_title("训练集图像")
    axes[0].axis('off')
    
    axes[1].imshow(np.array(img2))
    axes[1].set_title("验证集图像")
    axes[1].axis('off')
    
    # 添加相似度信息
    plt.suptitle(f"相似度: {similarity:.4f}", fontsize=16)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查训练集和验证集中的相似图像')
    parser.add_argument('--train_dir', default='aircraft_dataset/train', type=str, help='训练集目录')
    parser.add_argument('--val_dir', default='aircraft_dataset/val', type=str, help='验证集目录')
    parser.add_argument('--output_dir', default='dataset-check-similar', type=str, help='输出目录')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--threshold', type=float, default=0.95, help='相似度阈值(0-1之间)')
    
    args = parser.parse_args()
    
    find_similar_images(args.train_dir, args.val_dir, args.output_dir, 
                        batch_size=args.batch_size, 
                        similarity_threshold=args.threshold) 