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

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")

# 参数设置
parser = argparse.ArgumentParser(description='飞行器图像分类CNN')
parser.add_argument('--batch_size', default=64, type=int, help='批处理大小')
parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
parser.add_argument('--lr', default=1e-3, type=float, help='学习率')
parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集路径')
parser.add_argument('--save_dir', default='models', type=str, help='模型保存路径')
parser.add_argument('--workers', default=4, type=int, help='数据加载器的线程数')
parser.add_argument('--gpu', default=0, type=int, help='指定使用的GPU编号')
args = parser.parse_args()

# 设置随机种子以便结果可复现
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # 设置cudnn为确定性模式，牺牲一点性能但确保结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 在多GPU情况下启用cuDNN自动优化
    torch.backends.cudnn.enabled = True

# 确保模型保存目录存在
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# 设备配置
if torch.cuda.is_available():
    device = torch.device(f"cuda:{args.gpu}")  # 可以指定特定GPU
else:
    device = torch.device("cpu")
print(f"使用设备: {device}")

# 数据预处理和增强
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
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
        label = self.labels[idx]

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label


# 定义CNN模型
class AircraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AircraftCNN, self).__init__()

        # 第一个卷积块 - 输入: 3x140x140, 输出: 64x70x70
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 输出: 32x140x140
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 输出: 64x140x140
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 64x70x70
        )

        # 第二个卷积块 - 输入: 64x70x70, 输出: 128x35x35
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 输出: 128x70x70
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 输出: 128x70x70
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 128x35x35
        )

        # 第三个卷积块 - 输入: 128x35x35, 输出: 256x17x17
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 输出: 256x35x35
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 输出: 256x35x35
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 256x17x17
        )

        # 第四个卷积块 - 输入: 256x17x17, 输出: 512x8x8
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # 输出: 512x17x17
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # 输出: 512x17x17
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出: 512x8x8
        )

        # 自适应池化层，确保输出大小固定为4x4，不管输入尺寸
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 输出: 512x4x4

        # 分类器部分 - 全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 输出: 512*4*4 = 8192
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 混合精度训练
try:
    from torch.cuda.amp import autocast, GradScaler

    use_amp = torch.cuda.is_available()  # 仅在CUDA可用时使用混合精度
    print(f"启用自动混合精度训练: {use_amp}")
except ImportError:
    use_amp = False
    print("PyTorch版本不支持自动混合精度训练")


# 训练函数
def train(model, train_loader, criterion, optimizer, epoch, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        # 清零梯度
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            # 使用混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # 缩放梯度并执行反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 打印训练状态
        if (batch_idx + 1) % 10 == 0:
            batch_time = time.time() - start_time
            print(f'Epoch: [{epoch + 1}/{args.epochs}][{batch_idx + 1}/{len(train_loader)}] '
                  f'Loss: {running_loss / (batch_idx + 1):.4f} '
                  f'Acc: {100. * correct / total:.2f}% ({correct}/{total}) '
                  f'Time: {batch_time:.1f}s')
            start_time = time.time()  # 重置时间

    return running_loss / len(train_loader), 100. * correct / total


# 测试函数
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_predicted = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 收集所有预测和目标用于混淆矩阵
            all_targets.extend(targets.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    print(f'Test Loss: {test_loss / len(test_loader):.4f} Acc: {100. * correct / total:.2f}% ({correct}/{total})')

    return test_loss / len(test_loader), 100. * correct / total, all_targets, all_predicted


# 可视化训练过程
def plot_metrics(train_losses, train_accs, test_losses, test_accs, save_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy over epochs')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# 绘制混淆矩阵
def plot_confusion_matrix(all_targets, all_predicted, class_names, save_path):
    cm = confusion_matrix(all_targets, all_predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # 数据加载
    train_dataset = AircraftDataset(root_dir=args.data_dir, subset='train', transform=train_transform)
    test_dataset = AircraftDataset(root_dir=args.data_dir, subset='test', transform=test_transform)

    # 使用更多workers来加速数据加载
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True  # 这有助于加速数据传输到GPU
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 类别名称
    class_names = train_dataset.classes
    print(f"类别: {class_names}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 初始化模型
    model = AircraftCNN(num_classes=len(class_names)).to(device)

    # 如果有多GPU，可以使用DataParallel
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    # 记录模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 学习率调度器，使用余弦退火调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 初始化混合精度训练的scaler
    scaler = GradScaler() if use_amp else None

    # 记录训练和测试指标
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    # 开始训练
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.epochs):
        # 训练阶段
        epoch_start = time.time()
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch, device, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 测试阶段
        test_loss, test_acc, all_targets, all_predicted = test(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 更新学习率
        scheduler.step()

        # 如果是最佳模型，保存模型
        if test_acc > best_acc:
            best_acc = test_acc
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'Saved best model with accuracy: {best_acc:.2f}%')

        # 每个epoch都保存一个检查点
        if isinstance(model, nn.DataParallel):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_accs': test_accs,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        else:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'train_accs': train_accs,
                'test_losses': test_losses,
                'test_accs': test_accs,
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # 绘制当前指标
        plot_metrics(train_losses, train_accs, test_losses, test_accs,
                     os.path.join(args.save_dir, 'training_metrics.png'))

        # 每5个epoch显示一次混淆矩阵
        if (epoch + 1) % 5 == 0:
            plot_confusion_matrix(all_targets, all_predicted, class_names,
                                  os.path.join(args.save_dir, f'confusion_matrix_epoch_{epoch + 1}.png'))

        # 打印每个epoch的时间
        epoch_time = time.time() - epoch_start
        print(f'Epoch {epoch + 1} 完成! 耗时: {epoch_time:.2f}秒')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')

    # 训练结束，计算总时间
    total_time = time.time() - start_time
    print(f'训练完成! 总用时: {total_time / 60:.2f}分钟')
    print(f'最佳准确率: {best_acc:.2f}%')

    # 最终评估
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    else:
        model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pth')))
    _, _, all_targets, all_predicted = test(model, test_loader, criterion, device)

    # 生成分类报告
    report = classification_report(all_targets, all_predicted, target_names=class_names)
    print("\n分类报告:")
    print(report)

    with open(os.path.join(args.save_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # 最终混淆矩阵
    plot_confusion_matrix(all_targets, all_predicted, class_names,
                          os.path.join(args.save_dir, 'final_confusion_matrix.png'))


if __name__ == '__main__':
    main()