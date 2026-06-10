import os
import shutil
from tqdm import tqdm
import argparse

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='将验证集合并回训练集')
    parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集根目录')
    args = parser.parse_args()
    
    # 基础路径
    base_dir = args.data_dir
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # 检查验证集目录是否存在
    if not os.path.exists(val_dir):
        print(f"验证集目录 {val_dir} 不存在，无需合并。")
        return
    
    # 获取验证集中的类别
    val_classes = os.listdir(val_dir)
    
    # 统计要移动的文件数量
    total_files = 0
    for class_name in val_classes:
        class_dir = os.path.join(val_dir, class_name)
        if os.path.isdir(class_dir):
            total_files += len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"开始将 {total_files} 张验证集图片合并回训练集...")
    
    # 遍历验证集中的每个类别
    moved_count = 0
    for class_name in val_classes:
        val_class_dir = os.path.join(val_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        
        # 确保训练集中对应的类别目录存在
        if not os.path.exists(train_class_dir):
            os.makedirs(train_class_dir)
        
        # 获取验证集中该类别的所有图片
        val_images = [f for f in os.listdir(val_class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # 移动每张图片到训练集
        for img_name in tqdm(val_images, desc=f"正在处理 {class_name}"):
            val_img_path = os.path.join(val_class_dir, img_name)
            train_img_path = os.path.join(train_class_dir, img_name)
            
            # 移动图片
            shutil.move(val_img_path, train_img_path)
            moved_count += 1
    
    # 删除空的验证集目录
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) == 0:
        shutil.rmtree(val_dir)
        print(f"已删除空的验证集目录 {val_dir}")
    
    print(f"合并完成! 共将 {moved_count} 张图片从验证集移回训练集。")
    
    # 统计合并后的训练集大小
    train_count = 0
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            class_count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            train_count += class_count
            print(f"类别 {class_name}: {class_count} 张图片")
    
    print(f"合并后训练集共有 {train_count} 张图片")

if __name__ == "__main__":
    main() 