import os
import random
import shutil
from tqdm import tqdm
import argparse

def main():
    # 参数解析
    parser = argparse.ArgumentParser(description='扩充飞行器数据集测试集')
    parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集根目录')
    parser.add_argument('--samples_per_class', default=450, type=int, help='每个类别从训练集移动到测试集的图片数量')
    parser.add_argument('--seed', default=42, type=int, help='随机种子，确保结果可复现')
    args = parser.parse_args()
    
    # 设置随机种子确保可复现
    random.seed(args.seed)
    
    # 基础路径
    base_dir = args.data_dir
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    
    # 确保测试目录存在
    os.makedirs(test_dir, exist_ok=True)
    
    # 处理每个类别
    total_moved = 0
    for class_name in sorted(os.listdir(train_dir)):
        class_train_dir = os.path.join(train_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        
        # 确保测试集类别目录存在
        os.makedirs(class_test_dir, exist_ok=True)
        
        # 获取训练集中该类别的所有图片
        all_images = [f for f in os.listdir(class_train_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        print(f"类别 {class_name} 训练集中有 {len(all_images)} 张图片")
        
        # 获取测试集中该类别已有的图片数量
        existing_test_images = [f for f in os.listdir(class_test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        existing_count = len(existing_test_images)
        print(f"类别 {class_name} 测试集中已有 {existing_count} 张图片")
        
        # 计算需要移动的图片数量，确保测试集达到目标数量
        target_count = 500  # 目标每类测试集样本数
        move_count = min(args.samples_per_class, max(0, target_count - existing_count))
        
        if move_count <= 0:
            print(f"类别 {class_name} 测试集已达到或超过目标数量，无需移动")
            continue
            
        if move_count > len(all_images):
            print(f"警告: 类别 {class_name} 训练集图片不足，只能移动 {len(all_images)} 张")
            move_count = len(all_images)
        
        # 随机选择要移动的图片
        images_to_move = random.sample(all_images, move_count)
        print(f"将移动 {len(images_to_move)} 张图片到测试集")
        
        # 移动图片
        for img in tqdm(images_to_move, desc=f"移动 {class_name} 图片"):
            src_path = os.path.join(class_train_dir, img)
            dst_path = os.path.join(class_test_dir, img)
            shutil.move(src_path, dst_path)
            total_moved += 1
        
        # 统计移动后的数量
        remaining_train = len(os.listdir(class_train_dir))
        moved_test = len(os.listdir(class_test_dir))
        print(f"类别 {class_name}: 训练集剩余 {remaining_train} 张，测试集现有 {moved_test} 张")
    
    print(f"\n数据集重构完成! 共移动了 {total_moved} 张图片到测试集")
    
    # 打印最终统计信息
    print("\n最终数据集统计:")
    total_train = 0
    total_test = 0
    
    for class_name in sorted(os.listdir(train_dir)):
        train_count = len([f for f in os.listdir(os.path.join(train_dir, class_name)) if f.endswith(('.jpg', '.jpeg', '.png'))])
        test_count = len([f for f in os.listdir(os.path.join(test_dir, class_name)) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"类别 {class_name}: 训练集 {train_count} 张，测试集 {test_count} 张")
        total_train += train_count
        total_test += test_count
    
    print(f"\n总计: 训练集 {total_train} 张，测试集 {total_test} 张，总数据量 {total_train + total_test} 张")

if __name__ == '__main__':
    main() 