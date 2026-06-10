import os
import random
import shutil
from tqdm import tqdm
import argparse

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='从训练集创建验证集')
    parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集根目录')
    parser.add_argument('--samples_per_class', default=500, type=int, help='每个类别随机抽取的验证集样本数')
    parser.add_argument('--seed', default=42, type=int, help='随机种子，确保结果可复现')
    parser.add_argument('--copy_mode', action='store_true', help='使用复制模式而不是移动模式（默认为移动模式，确保不重复）')
    args = parser.parse_args()
    
    # 设置随机种子确保可复现
    random.seed(args.seed)
    
    # 基础路径
    base_dir = args.data_dir
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    # 确保验证集目录存在
    os.makedirs(val_dir, exist_ok=True)
    
    # 总共移动的图片数量
    total_moved = 0
    
    # 处理每个类别
    operation_name = "复制" if args.copy_mode else "移动"
    print(f"开始从训练集创建验证集，每类{operation_name} {args.samples_per_class} 张图片")
    classes = sorted(os.listdir(train_dir))
    
    for class_name in classes:
        print(f"\n处理类别: {class_name}")
        
        # 训练集中该类的图片目录
        class_train_dir = os.path.join(train_dir, class_name)
        
        # 创建验证集中对应类别的目录
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)
        
        # 获取训练集中该类别的所有图片
        all_images = [f for f in os.listdir(class_train_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
        
        print(f"  训练集中有 {len(all_images)} 张图片")
        
        # 确定要移动的图片数量
        move_count = min(args.samples_per_class, len(all_images))
        if move_count < args.samples_per_class:
            print(f"  警告: 类别 {class_name} 中图片不足 {args.samples_per_class} 张，只能{operation_name} {move_count} 张")
        
        # 随机选择图片
        images_to_move = random.sample(all_images, move_count)
        
        # 移动选中的图片到验证集
        print(f"  {operation_name} {move_count} 张图片到验证集...")
        for img_name in tqdm(images_to_move, desc=f"  {operation_name} {class_name} 图片"):
            src_path = os.path.join(class_train_dir, img_name)
            dst_path = os.path.join(class_val_dir, img_name)
            
            if args.copy_mode:
                shutil.copy(src_path, dst_path)  # 复制模式（会导致训练集和验证集重复）
            else:
                shutil.move(src_path, dst_path)  # 移动模式（确保不重复）
            
            total_moved += 1
    
    # 输出验证集统计信息
    val_classes = os.listdir(val_dir)
    val_stats = {}
    total_val_images = 0
    
    for cls in val_classes:
        cls_dir = os.path.join(val_dir, cls)
        image_count = len([f for f in os.listdir(cls_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        val_stats[cls] = image_count
        total_val_images += image_count
    
    print("\n验证集创建完成!")
    print(f"总共{operation_name}了 {total_val_images} 张验证集图片")
    print("验证集类别分布:")
    for cls, count in val_stats.items():
        print(f"  {cls}: {count} 张图片")
    
    # 现在重新统计训练集
    train_stats = {}
    total_train_images = 0
    
    for cls in classes:
        cls_dir = os.path.join(train_dir, cls)
        image_count = len([f for f in os.listdir(cls_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))])
        train_stats[cls] = image_count
        total_train_images += image_count
    
    print("\n更新后的训练集统计:")
    print(f"总共剩余 {total_train_images} 张训练集图片")
    print("训练集类别分布:")
    for cls, count in train_stats.items():
        print(f"  {cls}: {count} 张图片")
    
    # 输出使用验证集的提示
    print("\n使用说明:")
    print("1. 在train-with-val.py中已经配置好验证集加载:")
    print("   val_dataset = AircraftDataset(args.data_dir, mode='val', transform=val_transform)")
    print("   val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)")
    print("2. 直接运行train-with-val.py使用验证集进行训练:")
    print("   python train-with-val.py")
    print("\n注意: 现在训练集和验证集是完全分离的，没有重复图片!")

if __name__ == "__main__":
    main() 