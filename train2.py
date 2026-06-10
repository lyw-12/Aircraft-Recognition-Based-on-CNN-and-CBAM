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
parser = argparse.ArgumentParser(description='飞行器图像分类ResNet改进版')
parser.add_argument('--batch_size', default=64, type=int, help='批处理大小')
parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集路径')
parser.add_argument('--save_dir', default='models2', type=str, help='模型保存路径')
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
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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


# 基本残差块
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


# ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=4):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        self._initialize_weights()

    def _make_layer(self, block, channels, blocks, stride=1):
        downsample = None

        # 如果步长不为1或输入通道数不等于输出通道数*扩展系数，则需要下采样
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(channels * block.expansion)
            )

        layers = []
        # 第一个残差块可能需要下采样
        layers.append(block(self.in_channels, channels, stride, downsample))

        # 更新输入通道数
        self.in_channels = channels * block.expansion

        # 添加剩余残差块
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes=4):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


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
    model = resnet18(num_classes=len(train_dataset.classes))
    model = model.to(device)

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

        # 打印测试结果
        print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.2f}% ({correct}/{total})")

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
            plt.savefig(os.path.join(args.save_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))
            plt.close()

            # 打印分类报告
            report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
            print(f"\n分类报告:\n{report}")

            # 保存分类报告到文件
            with open(os.path.join(args.save_dir, f'classification_report_epoch_{epoch + 1}.txt'), 'w') as f:
                f.write(report)

        # 保存当前模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss,
            'acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc
        }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # 打印当前epoch的总结
        print(f"Epoch {epoch + 1} 完成! 耗时: {(time.time() - epoch_start_time):.2f}秒")
        print(f"当前学习率: {current_lr:.6f}")

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"训练完成! 总用时: {total_time / 60:.2f}分钟")
    print(f"最佳准确率: {best_acc:.2f}% (Epoch {best_epoch + 1})")

    # 加载最佳模型权重并在测试集上进行最终评估
    model.load_state_dict(best_model_wts)
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

    # 计算测试集上的最终损失和准确率
    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    # 打印最终测试结果
    print(f"最终测试结果: Loss: {test_loss:.4f} Acc: {test_acc:.2f}% ({correct}/{total})")

    # 生成并保存最终混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.title(f'最终混淆矩阵 (Acc: {test_acc:.2f}%)')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'final_confusion_matrix.png'))

    # 打印分类报告
    report = classification_report(all_labels, all_preds, target_names=train_dataset.classes)
    print(f"\n最终分类报告:\n{report}")

    # 保存分类报告到文件
    with open(os.path.join(args.save_dir, 'final_classification_report.txt'), 'w') as f:
        f.write(report)

    # 绘制训练和测试损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')

    # 绘制训练和测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(test_accs, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
    plt.close()

    # 返回最佳准确率和相应的epoch
    return best_acc, best_epoch


if __name__ == '__main__':
    best_acc, best_epoch = train_and_evaluate()
    print(f"最佳模型准确率: {best_acc:.2f}% (Epoch {best_epoch + 1})")