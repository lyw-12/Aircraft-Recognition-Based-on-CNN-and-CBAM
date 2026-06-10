import os
import argparse
from collections import defaultdict

def count_images(directory, extensions=('.jpg', '.jpeg', '.png')):
    """统计指定目录中的图片数量"""
    if not os.path.exists(directory):
        return 0
    
    return len([f for f in os.listdir(directory) if f.lower().endswith(extensions)])

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算数据集中每个类别的图片数量')
    parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集根目录')
    parser.add_argument('--detailed', action='store_true', help='是否显示详细统计信息')
    args = parser.parse_args()
    
    # 导入tabulate
    try:
        from tabulate import tabulate
    except ImportError:
        print("需要安装tabulate库。正在尝试安装...")
        import subprocess
        try:
            subprocess.check_call(["pip", "install", "tabulate"])
            print("tabulate库安装成功!")
            from tabulate import tabulate
        except:
            print("无法安装tabulate库。请手动运行: pip install tabulate")
            # 使用简单的表格输出作为备选
            def simple_tabulate(data, headers):
                result = []
                # 计算每列的最大宽度
                col_widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
                
                # 添加表头
                header_row = ' | '.join(f"{h:{w}s}" for h, w in zip(headers, col_widths))
                result.append(header_row)
                result.append('-' * len(header_row))
                
                # 添加数据行
                for row in data:
                    result.append(' | '.join(f"{str(cell):{w}s}" for cell, w in zip(row, col_widths)))
                
                return '\n'.join(result)
            
            tabulate = simple_tabulate
    
    dataset_dir = args.data_dir
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    
    # 检查目录是否存在
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"错误: 训练集或测试集目录不存在于 {dataset_dir}")
        return
    
    # 获取所有类别
    categories = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    if not categories:
        print(f"错误: 在 {train_dir} 中没有找到类别目录")
        return
    
    print(f"数据集路径: {os.path.abspath(dataset_dir)}")
    print(f"发现 {len(categories)} 个类别: {', '.join(categories)}")
    
    # 统计每个类别的图片数量
    stats = []
    total_train = 0
    total_test = 0
    
    for category in categories:
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        
        train_count = count_images(train_category_dir)
        test_count = count_images(test_category_dir)
        
        total_train += train_count
        total_test += test_count
        
        stats.append([category, train_count, test_count, train_count + test_count])
    
    # 添加总计行
    stats.append(["总计", total_train, total_test, total_train + total_test])
    
    # 打印表格
    headers = ["类别", "训练集数量", "测试集数量", "总数量"]
    
    if 'simple_tabulate' in locals():
        print("\n" + tabulate(stats, headers))
    else:
        print("\n" + tabulate(stats, headers=headers, tablefmt="grid"))
    
    # 计算比例
    train_percent = (total_train / (total_train + total_test)) * 100
    test_percent = (total_test / (total_train + total_test)) * 100
    print(f"\n训练集占比: {train_percent:.2f}%, 测试集占比: {test_percent:.2f}%")
    
    # 如果要求详细信息，显示每个类别的文件列表
    if args.detailed:
        print("\n详细信息:")
        for category in categories:
            train_category_dir = os.path.join(train_dir, category)
            test_category_dir = os.path.join(test_dir, category)
            
            # 获取训练集图片列表
            if os.path.exists(train_category_dir):
                train_files = sorted([f for f in os.listdir(train_category_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            else:
                train_files = []
            
            # 获取测试集图片列表
            if os.path.exists(test_category_dir):
                test_files = sorted([f for f in os.listdir(test_category_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            else:
                test_files = []
            
            print(f"\n类别: {category}")
            print(f"  训练集图片 ({len(train_files)}):")
            for i, file in enumerate(train_files):
                if i < 5:  # 只显示前5张图片
                    print(f"    - {file}")
                elif i == 5:
                    print(f"    - ... 还有 {len(train_files) - 5} 张图片")
                else:
                    break
            
            print(f"  测试集图片 ({len(test_files)}):")
            for i, file in enumerate(test_files):
                if i < 5:  # 只显示前5张图片
                    print(f"    - {file}")
                elif i == 5:
                    print(f"    - ... 还有 {len(test_files) - 5} 张图片")
                else:
                    break
    
    # 检查是否有文件名重复
    print("\n检查训练集和测试集是否存在同名文件...")
    duplicates_by_name = defaultdict(list)
    
    for category in categories:
        train_category_dir = os.path.join(train_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        
        if os.path.exists(train_category_dir) and os.path.exists(test_category_dir):
            train_files = set([f for f in os.listdir(train_category_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            test_files = set([f for f in os.listdir(test_category_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            # 找出同名文件
            common_files = train_files.intersection(test_files)
            if common_files:
                duplicates_by_name[category] = common_files
    
    if any(duplicates_by_name.values()):
        print("\n警告: 发现训练集和测试集中存在同名文件!")
        for category, files in duplicates_by_name.items():
            print(f"  类别 {category}: {len(files)} 个同名文件")
            for i, file in enumerate(sorted(files)):
                if i < 5:
                    print(f"    - {file}")
                elif i == 5:
                    print(f"    - ... 还有 {len(files) - 5} 个文件")
                else:
                    break
        print("\n文件名相同不一定意味着内容相同，但建议检查它们是否为重复图片。")
    else:
        print("未发现训练集和测试集间存在同名文件，这是个好兆头!")

if __name__ == "__main__":
    main() 