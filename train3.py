import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import argparse
from torch.cuda.amp import autocast, GradScaler

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")

# 参数设置
parser = argparse.ArgumentParser(description='飞行器图像分类简化CNN模型')
parser.add_argument('--batch_size', default=64, type=int, help='批处理大小')
parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集路径')
parser.add_argument('--save_dir', default='models3', type=str, help='模型保存路径')
parser.add_argument('--workers', default=4, type=int, help='数据加载器的线程数')
parser.add_argument('--gpu', default=0, type=int, help='指定使用的GPU编号')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='权重衰减系数')
args = parser.parse_args()

# 设置随机种子以便结果可复现
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

# 确保模型保存目录存在
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# 设备配置
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# 启用自动混合精度训练
use_amp = torch.cuda.is_available()
print(f"启用自动混合精度训练: {use_amp}")

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 自定义数据集类
class AircraftDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        self.root_dir = os.path.join(root_dir, subset)
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.images = []
        self.labels = []

        # 加载所有图像和标签
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# 简化的残差块
class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SimpleResidualBlock, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接，如果输入输出通道数不同，需要1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 应用残差连接
        out = out + self.shortcut(residual)

        out = self.relu(out)
        return out


# 通道注意力机制模块 - 创新点1：轻量级注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(avg_pool).view(b, c, 1, 1)
        return x * channel_attention


# 简化的CNN模型 - 比ResNet18更轻量，比improved_aircraft_cnn更简单
class SimplifiedAircraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimplifiedAircraftCNN, self).__init__()

        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        # 第一个残差块组 - 简化为只有1个残差块
        self.layer1 = nn.Sequential(
            SimpleResidualBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 64x70x70
        )

        # 第二个残差块组 - 简化为只有1个残差块
        self.layer2 = nn.Sequential(
            SimpleResidualBlock(64, 128, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 128x17x17
        )

        # 第三个残差块组 - 简化为只有1个残差块
        self.layer3 = nn.Sequential(
            SimpleResidualBlock(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 256x8x8
        )

        # 通道注意力机制 - 创新点1：只在关键位置添加注意力机制
        self.attention = ChannelAttention(256)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 输出: 256x4x4

        # 创新点2：特征聚合 - 简化版，不使用双路径结构
        self.feature_aggregation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),  # 使用较小的dropout比例
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 应用通道注意力 - 创新点1
        x = self.attention(x)

        x = self.adaptive_pool(x)
        x = self.feature_aggregation(x)

        return x


def train_and_evaluate():
    # 载入数据集
    train_dataset = AircraftDataset(root_dir=args.data_dir, subset='train', transform=train_transform)
    test_dataset = AircraftDataset(root_dir=args.data_dir, subset='test', transform=test_transform)

    # 查看类别分布情况
    train_counts = {}
    for _, label in train_dataset:
        if label in train_counts:
            train_counts[label] += 1
        else:
            train_counts[label] = 1

    print(f"训练集分布: {train_counts}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型信息目录
    model_info_dir = os.path.join(args.save_dir, 'model_info')
    os.makedirs(model_info_dir, exist_ok=True)

    # 保存模型架构信息
    with open(os.path.join(model_info_dir, 'model_architecture.txt'), 'w') as f:
        f.write("SimplifiedAircraftCNN 模型架构\n")
        f.write("=" * 50 + "\n\n")
        f.write("特点:\n")
        f.write("1. 轻量级残差网络结构，每层仅包含1个残差块\n")
        f.write("2. 采用通道注意力机制，增强特征表达能力\n")
        f.write("3. 简化的特征聚合结构，减少参数量\n\n")
        f.write("主要创新点:\n")
        f.write("1. 通道注意力机制: 帮助模型关注重要特征通道\n")
        f.write("2. 特征聚合优化: 在保持性能的同时减少计算复杂度\n\n")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 创建模型
    model = SimplifiedAircraftCNN(num_classes=len(train_dataset.classes))
    model = model.to(device)

    # 打印模型结构和参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(model)

    # 保存模型结构和参数信息
    with open(os.path.join(model_info_dir, 'model_parameters.txt'), 'w') as f:
        f.write(f"模型总参数数量: {total_params:,}\n")
        f.write(f"可训练参数数量: {trainable_params:,}\n\n")
        f.write("模型结构详情:\n")
        f.write(str(model))

    # 记录训练配置
    with open(os.path.join(model_info_dir, 'training_config.txt'), 'w') as f:
        f.write(f"批处理大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"初始学习率: {args.lr}\n")
        f.write(f"权重衰减系数: {args.weight_decay}\n")
        f.write(f"优化器: AdamW (amsgrad=True)\n")
        f.write(f"学习率调度: CosineAnnealingLR (T_max={args.epochs}, eta_min=1e-6)\n")
        f.write(f"数据集路径: {args.data_dir}\n")
        f.write(f"训练设备: {device}\n")
        f.write(f"是否使用混合精度训练: {use_amp}\n")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=True
    )

    # 学习率调度器 - 余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    # 混合精度训练
    scaler = GradScaler(enabled=use_amp)

    # 存储训练和测试的损失与准确率
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    # 创建训练日志文件
    training_log_path = os.path.join(args.save_dir, 'training_log.csv')
    with open(training_log_path, 'w') as f:
        f.write("epoch,train_loss,train_acc,test_loss,test_acc,learning_rate,epoch_time\n")

    # 追踪最佳准确率和模型
    best_acc = 0.0
    best_model_wts = None
    best_epoch = 0

    # 开始训练
    start_time = time.time()

    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 零化梯度
            optimizer.zero_grad()

            # 使用混合精度
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 更新参数
            scaler.step(optimizer)
            scaler.update()

            # 更新统计信息
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 每10个批次打印一次信息
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"Epoch: [{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(train_loader)}] "
                      f"Loss: {train_loss / (batch_idx + 1):.4f} "
                      f"Acc: {100. * correct / total:.2f}% ({correct}/{total}) "
                      f"Time: {(time.time() - epoch_start_time):.1f}s", flush=True)

        # 更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 计算训练集上的平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 在测试集上评估
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                with autocast(enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 收集预测和标签用于计算混淆矩阵
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算测试集上的平均损失和准确率
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 计算本轮训练时间
        epoch_time = time.time() - epoch_start_time

        # 打印测试结果
        print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% ({correct}/{total})")

        # 更新训练日志
        with open(training_log_path, 'a') as f:
            f.write(
                f"{epoch + 1},{train_loss:.6f},{train_acc:.2f},{test_loss:.6f},{test_acc:.2f},{current_lr:.6f},{epoch_time:.2f}\n")

        # 为当前epoch创建目录
        epoch_dir = os.path.join(args.save_dir, f'epoch_{epoch + 1}')
        os.makedirs(epoch_dir, exist_ok=True)

        # 如果当前模型是最好的，保存它
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = model.state_dict()
            best_epoch = epoch
            # 保存最佳模型
            torch.save(best_model_wts, os.path.join(args.save_dir, 'best_model.pth'))
            # 生成并保存最佳模型的混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
            plt.title(f'Confusion Matrix - Epoch {epoch + 1} (Acc: {test_acc:.2f}%)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, f'best_confusion_matrix.png'))
            plt.savefig(os.path.join(epoch_dir, f'confusion_matrix.png'))
            plt.close()

            # 打印分类报告
            report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
            print(f"\n分类报告:\n{report}")

            # 保存分类报告到最佳模型目录和当前epoch目录
            with open(os.path.join(args.save_dir, f'best_classification_report.txt'), 'w') as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Accuracy: {test_acc:.2f}%\n\n")
                f.write(report)

            with open(os.path.join(epoch_dir, f'classification_report.txt'), 'w') as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Accuracy: {test_acc:.2f}%\n\n")
                f.write(report)

            # 标记为最佳模型
            with open(os.path.join(epoch_dir, 'is_best_model.txt'), 'w') as f:
                f.write(f"测试准确率: {test_acc:.2f}%\n")
                f.write(f"这是当前最佳模型 (Epoch {epoch + 1})")
        else:
            # 生成并保存当前epoch的混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
            plt.title(f'Confusion Matrix - Epoch {epoch + 1} (Acc: {test_acc:.2f}%)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(epoch_dir, f'confusion_matrix.png'))
            plt.close()

            # 保存分类报告到当前epoch目录
            report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
            with open(os.path.join(epoch_dir, f'classification_report.txt'), 'w') as f:
                f.write(f"Epoch: {epoch + 1}\n")
                f.write(f"Accuracy: {test_acc:.2f}%\n\n")
                f.write(report)

        # 保存当前模型
        checkpoint_path = os.path.join(epoch_dir, 'checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }, checkpoint_path)

        # 保存当前epoch的详细信息
        with open(os.path.join(epoch_dir, 'epoch_info.txt'), 'w') as f:
            f.write(f"Epoch: {epoch + 1}/{args.epochs}\n")
            f.write(f"训练损失: {train_loss:.6f}\n")
            f.write(f"训练准确率: {train_acc:.2f}%\n")
            f.write(f"测试损失: {test_loss:.6f}\n")
            f.write(f"测试准确率: {test_acc:.2f}%\n")
            f.write(f"学习率: {current_lr:.6f}\n")
            f.write(f"耗时: {epoch_time:.2f}秒\n")

        # 打印当前epoch的总结
        print(f"Epoch {epoch + 1} 完成! 耗时: {epoch_time:.2f}秒")
        print(f"当前学习率: {current_lr:.6f}")

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总用时: {total_time / 60:.2f}分钟")
    print(f"最佳准确率: {best_acc:.2f}% (Epoch {best_epoch + 1})")

    # 保存训练总结
    with open(os.path.join(args.save_dir, 'training_summary.txt'), 'w') as f:
        f.write(f"训练总结\n")
        f.write(f"=" * 30 + "\n\n")
        f.write(f"总训练时间: {total_time / 60:.2f}分钟\n")
        f.write(f"最佳准确率: {best_acc:.2f}% (Epoch {best_epoch + 1})\n")
        f.write(f"总参数数量: {total_params:,}\n")
        f.write(f"模型结构: SimplifiedAircraftCNN\n")
        f.write(f"创新点:\n")
        f.write(f"1. 通道注意力机制\n")
        f.write(f"2. 轻量级特征聚合\n")

    return model, best_acc


if __name__ == "__main__":
    print("=" * 50)
    print("开始训练简化版飞行器识别CNN模型...")
    print(f"模型将保存到: {args.save_dir}")
    print(f"训练轮数: {args.epochs}")
    print(f"批处理大小: {args.batch_size}")
    print(f"初始学习率: {args.lr}")
    print("=" * 50)

    model, best_acc = train_and_evaluate()

    print("=" * 50)
    print(f"模型训练完成，最佳准确率: {best_acc:.2f}%")
    print("=" * 50)