# 数据集重复图片检查工具

## 功能介绍

这个工具用于检查训练集和验证集中是否存在重复的图片。当数据集划分不当时，可能会导致相同的图片同时出现在训练集和验证集中，这会影响模型评估的准确性，因为模型可能会在验证时"看到"它在训练中已经学习过的完全相同的图片。

工具使用图像哈希技术来检测重复，具体工作流程如下：

1. 扫描训练集和验证集中的所有图片
2. 对每张图片进行预处理（调整大小）并计算哈希值
3. 比较哈希值来识别重复图片
4. 生成详细报告
5. 将训练集和验证集之间的重复图片复制到指定目录

## 使用方法

### 基本用法

```bash
python check_duplicate_images.py
```

默认情况下，脚本会：
- 检查 `aircraft_dataset/train` 目录下的训练集图片
- 检查 `aircraft_dataset/val` 目录下的验证集图片
- 将结果保存到 `dataset-check` 目录

### 自定义参数

```bash
python check_duplicate_images.py --train_dir [训练集路径] --val_dir [验证集路径] --output_dir [输出目录] --workers [进程数]
```

参数说明：
- `--train_dir`：训练集目录路径（默认：`aircraft_dataset/train`）
- `--val_dir`：验证集目录路径（默认：`aircraft_dataset/val`）
- `--output_dir`：结果输出目录（默认：`dataset-check`）
- `--workers`：并行处理的进程数，默认使用系统可用的CPU核心数

### 示例

```bash
python check_duplicate_images.py --train_dir aircraft_dataset/train --val_dir aircraft_dataset/val --output_dir dataset-check --workers 4
```

## 输出结果

脚本会在输出目录中生成以下文件和目录：

1. `summary_report.txt`：总体摘要报告，包含检测到的重复图片数量和按类别统计
2. `duplicate_images_report.csv`：所有重复图片的详细信息
3. `train_val_duplicates.csv`：仅训练集和验证集之间的重复图片
4. `duplicate_class_stats.csv`：按类别统计的重复图片数量
5. `duplicates/`：包含训练集和验证集之间重复图片的副本，按组分类

## 工作原理

### 图像哈希

该工具使用感知哈希（Perceptual Hashing）技术来检测重复图片：

1. 将图片调整为统一大小（32×32像素）
2. 将图片转换为数字数组
3. 对数组计算MD5哈希值

这种方法可以检测到完全相同的图片，但对于稍微调整过（如轻微裁剪、旋转或色彩变化）的图片可能无法检测。

### 并行处理

工具使用Python的ProcessPoolExecutor进行并行处理，大大提高了处理大量图片的速度。并行处理会充分利用多核CPU，显著减少处理时间。

## 注意事项

1. 处理大量图片可能需要较长时间，请耐心等待
2. 使用并行处理会消耗较多内存，如遇内存不足，可以减少`--workers`参数的值
3. 如果数据集非常大，可能需要分批处理

## 如何处理重复图片

检测到重复图片后，您可以：

1. 从验证集中移除与训练集重复的图片
2. 重新划分数据集，确保训练集和验证集没有重叠
3. 如果重复数量较少，可以评估其对模型性能的影响是否显著 