import os
import argparse
from collections import defaultdict
import pandas as pd
from tabulate import tabulate


def count_images_in_directory(directory):
    """统计指定目录下各类别的图片数量"""
    if not os.path.exists(directory):
        print(f"警告：目录 {directory} 不存在")
        return None

    class_counts = defaultdict(int)
    total_count = 0

    # 常见图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif', '.webp'}

    # 遍历目录
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)

        # 检查是否是目录
        if os.path.isdir(class_dir):
            count = 0
            # 计算该类别下的图片数量
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
                    count += 1

            class_counts[class_name] = count
            total_count += count

    class_counts['总计'] = total_count
    return class_counts


def main():
    parser = argparse.ArgumentParser(description='统计数据集中的图片数量')
    parser.add_argument('--train_dir', type=str, default='aircraft_dataset/train', help='训练集目录路径')
    parser.add_argument('--test_dir', type=str, default='aircraft_dataset/test', help='测试集目录路径')
    parser.add_argument('--val_dir', type=str, default='aircraft_dataset/val', help='验证集目录路径')

    args = parser.parse_args()

    # 统计各集合的图片数量
    print("正在统计数据集图片数量...")

    train_counts = count_images_in_directory(args.train_dir)
    test_counts = count_images_in_directory(args.test_dir)
    val_counts = count_images_in_directory(args.val_dir)

    # 准备结果数据
    all_classes = set()
    if train_counts:
        all_classes.update(train_counts.keys())
    if test_counts:
        all_classes.update(test_counts.keys())
    if val_counts:
        all_classes.update(val_counts.keys())

    # 移除"总计"，稍后添加
    if '总计' in all_classes:
        all_classes.remove('总计')

    # 准备表格数据
    table_data = []
    for class_name in sorted(all_classes):
        if class_name != '总计':
            row = {
                '类别': class_name,
                '训练集': train_counts.get(class_name, 0) if train_counts else 0,
                '测试集': test_counts.get(class_name, 0) if test_counts else 0
            }
            if val_counts:
                row['验证集'] = val_counts.get(class_name, 0)

            table_data.append(row)

    # 添加总计行
    total_row = {
        '类别': '总计',
        '训练集': train_counts.get('总计', 0) if train_counts else 0,
        '测试集': test_counts.get('总计', 0) if test_counts else 0
    }
    if val_counts:
        total_row['验证集'] = val_counts.get('总计', 0)

    table_data.append(total_row)

    # 转换为DataFrame并打印
    df = pd.DataFrame(table_data)
    print("\n数据集统计结果:")
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))

    # 保存结果到CSV
    csv_file = 'dataset_statistics.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n统计结果已保存到 {csv_file}")


if __name__ == "__main__":
    main()