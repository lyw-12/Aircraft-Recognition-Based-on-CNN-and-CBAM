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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns
import argparse
import json
import pandas as pd
from datetime import datetime
import shutil
import multiprocessing
import torch.nn.functional as F

# 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"可用GPU数量: {torch.cuda.device_count()}")

# 参数设置
parser = argparse.ArgumentParser(description='飞行器图像分类第一次优化模型(ResNet+注意力机制)')
parser.add_argument('--batch_size', default=32, type=int, help='批处理大小')
parser.add_argument('--epochs', default=50, type=int, help='训练轮数')
parser.add_argument('--lr', default=5e-4, type=float, help='学习率')
parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集路径')
parser.add_argument('--save_dir', default='newmodels-resnet', type=str, help='模型保存路径')
parser.add_argument('--workers', default=0, type=int, help='数据加载器的线程数')
parser.add_argument('--gpu', default=0, type=int, help='指定使用的GPU编号')
parser.add_argument('--save_freq', default=5, type=int, help='每多少个epoch保存一次模型')
parser.add_argument('--eval_freq', default=1, type=int, help='每多少个epoch进行一次详细评估')
parser.add_argument('--patience', default=10, type=int, help='早停耐心值，连续多少个epoch验证集性能不提升就停止训练')
parser.add_argument('--use_validation', action='store_true', help='是否使用验证集(需要先创建)')
args = parser.parse_args()

# 创建保存目录和子目录
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for subdir in ['checkpoints', 'metrics', 'plots', 'results', 'logs']:
    os.makedirs(os.path.join(args.save_dir, subdir), exist_ok=True)

# 设置设备
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# 初始化日志文件
log_file = os.path.join(args.save_dir, 'logs', 'training_log.txt')
with open(log_file, 'w') as f:
    f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据集路径: {os.path.abspath(args.data_dir)}\n")
    f.write(f"训练设备: {device}\n")
    f.write(f"批量大小: {args.batch_size}\n")
    f.write(f"学习率: {args.lr}\n")
    f.write(f"计划训练轮数: {args.epochs}\n")
    f.write("-" * 50 + "\n\n")

# 记录函数
def log(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

# 数据预处理和增强
# 使用更多的数据增强
train_transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集和测试集使用相同的变换
val_transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = val_transform

# 定义数据集类
class AircraftDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # 加载数据集
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
        
        # 读取和转换图像
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label

# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # shortcut连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

# CBAM注意力模块
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

# 定义ResNet+注意力机制模型
class ResNetAttention(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetAttention, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # 注意力模块
        self.cbam = CBAM(256)
        
        # 自适应平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全连接分类器
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # 注意力机制
        x = self.cbam(x)
        
        # 全局池化
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # 分类器
        x = self.fc(x)
        
        return x 

# 创建数据加载器
def main():
    log("正在加载数据集...")
    train_dataset = AircraftDataset(args.data_dir, mode='train', transform=train_transform)
    test_dataset = AircraftDataset(args.data_dir, mode='test', transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 加载验证集(如果启用)
    val_loader = None
    if args.use_validation:
        try:
            val_dataset = AircraftDataset(args.data_dir, mode='val', transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            log(f"验证集样本数: {len(val_dataset)}")
        except Exception as e:
            log(f"警告: 验证集加载失败，原因: {str(e)}")
            log("请先运行dataset-val.py创建验证集，或不使用--use_validation参数")
            log("继续使用测试集进行验证...")
            args.use_validation = False

    # 获取类别名称
    class_names = train_dataset.classes
    num_classes = len(class_names)
    log(f"类别: {class_names}")
    log(f"训练集样本数: {len(train_dataset)}")
    log(f"测试集样本数: {len(test_dataset)}")

    # 保存数据集信息
    dataset_info = {
        "classes": class_names,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "train_samples_per_class": {cls: sum(1 for l in train_dataset.labels if l == i) 
                                  for i, cls in enumerate(class_names)},
        "test_samples_per_class": {cls: sum(1 for l in test_dataset.labels if l == i) 
                                 for i, cls in enumerate(class_names)}
    }

    if args.use_validation:
        dataset_info["val_samples"] = len(val_dataset)
        dataset_info["val_samples_per_class"] = {cls: sum(1 for l in val_dataset.labels if l == i) 
                                              for i, cls in enumerate(class_names)}

    with open(os.path.join(args.save_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=4)

    # 创建模型
    model = ResNetAttention(num_classes=num_classes)
    model = model.to(device)

    # 保存模型架构信息
    model_info = {
        'model_name': 'ResNetAttention',
        'num_params': sum(p.numel() for p in model.parameters()),
        'input_size': (3, 140, 140),
        'output_size': num_classes,
        'architecture': 'ResNet + CBAM注意力机制',
        'layers': str(model)
    }

    with open(os.path.join(args.save_dir, 'model_architecture.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # 训练函数
    def train(epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 打印批次信息
            if (batch_idx + 1) % 50 == 0 or batch_idx + 1 == len(train_loader):
                log(f'Epoch: {epoch + 1} | Batch: {batch_idx + 1}/{len(train_loader)} | Loss: {running_loss / (batch_idx + 1):.4f} | Acc: {100. * correct / total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_time = time.time() - start_time
        
        log(f'训练 Epoch: {epoch + 1} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Time: {train_time:.2f}s')
        
        return train_loss, train_acc

    # 验证函数
    def validate(epoch):
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_predicted = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集所有预测和目标用于生成混淆矩阵
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        
        # 计算准确率
        acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)
        log(f'\n验证 Epoch: {epoch + 1} | Loss: {val_loss:.4f} | Acc: {acc:.2f}%\n')
        
        # 保存分类报告
        report = classification_report(all_targets, all_predicted, target_names=class_names, digits=4)
        log(f"\n验证集分类报告 (Epoch {epoch+1}):\n{report}")
        
        # 每n个epoch生成一次详细评估报告
        if (epoch + 1) % args.eval_freq == 0:
            # 生成混淆矩阵
            cm = confusion_matrix(all_targets, all_predicted)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'验证集 Epoch {epoch + 1} 混淆矩阵')
            plt.savefig(os.path.join(args.save_dir, 'plots', f'val_confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()
            
            # 保存分类报告
            report_path = os.path.join(args.save_dir, 'results', f'val_classification_report_epoch_{epoch+1}.txt')
            with open(report_path, 'w') as f:
                f.write(report)
        
        return val_loss, acc, all_targets, all_predicted

    # 测试函数
    def test(epoch, is_final=False):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_targets = []
        all_predicted = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 统计
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集所有预测和目标用于生成混淆矩阵
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        
        # 计算准确率
        acc = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        log(f'\n测试 Epoch: {epoch + 1} | Loss: {test_loss:.4f} | Acc: {acc:.2f}%\n')
        
        # 保存分类报告
        report = classification_report(all_targets, all_predicted, target_names=class_names, digits=4)
        log(f"\n测试集分类报告 (Epoch {epoch+1}):\n{report}")
        
        # 生成详细测试评估
        if is_final or (epoch + 1) % args.eval_freq == 0:
            # 生成混淆矩阵
            cm = confusion_matrix(all_targets, all_predicted)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title(f'测试集 Epoch {epoch + 1} 混淆矩阵')
            
            # 保存混淆矩阵图
            if is_final:
                plt.savefig(os.path.join(args.save_dir, 'plots', 'final_confusion_matrix.png'))
            else:
                plt.savefig(os.path.join(args.save_dir, 'plots', f'test_confusion_matrix_epoch_{epoch+1}.png'))
            plt.close()
            
            # 保存分类报告
            report_path = os.path.join(args.save_dir, 'results', f'test_classification_report_epoch_{epoch+1}.txt')
            with open(report_path, 'w') as f:
                f.write(report)
        
        return test_loss, acc, all_targets, all_predicted

    # 记录训练历史
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    best_acc = 0
    best_epoch = 0
    patience_counter = 0  # 早停计数器

    # 创建CSV文件记录训练过程
    csv_file = os.path.join(args.save_dir, 'logs', 'training_history.csv')
    with open(csv_file, 'w') as f:
        if args.use_validation:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc,test_loss,test_acc,lr\n')
        else:
            f.write('epoch,train_loss,train_acc,test_loss,test_acc,lr\n')

    # 训练模型
    log("开始训练ResNet+注意力机制模型...")
    total_start_time = time.time()

    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_acc = train(epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证一个epoch (如果启用验证集)
        if args.use_validation:
            val_loss, val_acc, val_targets, val_predicted = validate(epoch)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
        
        # 测试一个epoch
        test_loss, test_acc, test_targets, test_predicted = test(epoch)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 记录到CSV
        with open(csv_file, 'a') as f:
            if args.use_validation:
                f.write(f'{epoch+1},{train_loss:.6f},{train_acc:.2f},{val_loss:.6f},{val_acc:.2f},{test_loss:.6f},{test_acc:.2f},{optimizer.param_groups[0]["lr"]:.6f}\n')
            else:
                f.write(f'{epoch+1},{train_loss:.6f},{train_acc:.2f},{test_loss:.6f},{test_acc:.2f},{optimizer.param_groups[0]["lr"]:.6f}\n')
        
        # 学习率调整 - 使用验证集或测试集
        if args.use_validation:
            monitor_acc = val_acc
            monitor_name = "验证集"
        else:
            monitor_acc = test_acc
            monitor_name = "测试集"
        
        scheduler.step(monitor_acc)
        
        # 保存最佳模型 (基于验证集或测试集准确率)
        if monitor_acc > best_acc:
            log(f'{monitor_name}准确率提升: {best_acc:.2f}% -> {monitor_acc:.2f}%')
            best_acc = monitor_acc
            best_epoch = epoch + 1
            patience_counter = 0  # 重置耐心计数器
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }, os.path.join(args.save_dir, 'checkpoints', 'best_model.pth'))
            
            # 保存最佳模型的分类报告
            if args.use_validation:
                report_path = os.path.join(args.save_dir, 'results', f'val_classification_report_epoch_{epoch+1}.txt')
                best_report_path = os.path.join(args.save_dir, 'results', 'best_val_classification_report.txt')
            else:
                report_path = os.path.join(args.save_dir, 'results', f'test_classification_report_epoch_{epoch+1}.txt')
                best_report_path = os.path.join(args.save_dir, 'results', 'best_test_classification_report.txt')
                
            shutil.copy(report_path, best_report_path)
        else:
            # 启用早停时，增加耐心计数器
            if args.use_validation and args.patience > 0:
                patience_counter += 1
                log(f'{monitor_name}准确率未提升，耐心计数: {patience_counter}/{args.patience}')
                
                # 早停检查
                if patience_counter >= args.patience:
                    log(f'提前停止训练! {monitor_name}准确率连续 {args.patience} 个epoch未提升')
                    break
        
        # 定期保存检查点
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'best_epoch': best_epoch
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoints', f'checkpoint_epoch_{epoch + 1}.pth'))

    # 训练结束，记录总时间
    total_time = time.time() - total_start_time
    log(f"训练结束! 总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    log(f"最佳{monitor_name}准确率: {best_acc:.2f}% (Epoch {best_epoch})")

    # 最终测试
    log('加载最佳模型进行最终测试...')
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoints', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc, _, _ = test(best_epoch - 1, is_final=True)
    log(f'最终测试准确率: {final_acc:.2f}%')

    # 保存训练结果信息
    accuracy_info = {
        "模型": "ResNet+注意力机制模型",
        "训练轮数": args.epochs if epoch+1 == args.epochs else epoch+1,  # 考虑早停
        "最佳Epoch": best_epoch,
        "最终测试精度": f"{final_acc:.2f}%",
        "模型参数总量": sum(p.numel() for p in model.parameters()),
        "训练集大小": len(train_dataset),
        "测试集大小": len(test_dataset),
        "总训练时间(秒)": total_time,
        "学习率": args.lr,
        "批量大小": args.batch_size,
    }

    if args.use_validation:
        accuracy_info["验证集大小"] = len(val_dataset)
        accuracy_info["最佳验证精度"] = f"{best_acc:.2f}%"
        accuracy_info["早停耐心值"] = args.patience
    else:
        accuracy_info["最佳测试精度"] = f"{best_acc:.2f}%"

    with open(os.path.join(args.save_dir, 'accuracy_info.txt'), 'w', encoding='utf-8') as f:
        f.write("ResNet+注意力机制模型准确率信息\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in accuracy_info.items():
            f.write(f"{key}: {value}\n")

    # 绘制训练曲线
    plt.figure(figsize=(12, 8))

    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
    if args.use_validation:
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'g-', label='验证损失')
    plt.plot(range(1, len(test_losses) + 1), test_losses, 'r-', label='测试损失')
    plt.axvline(x=best_epoch, color='gray', linestyle='--', label=f'最佳epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('训练、验证和测试损失')
    plt.grid(True)

    # 准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(range(1, len(train_accs) + 1), train_accs, 'b-', label='训练准确率')
    if args.use_validation:
        plt.plot(range(1, len(val_accs) + 1), val_accs, 'g-', label='验证准确率')
    plt.plot(range(1, len(test_accs) + 1), test_accs, 'r-', label='测试准确率')
    plt.axvline(x=best_epoch, color='gray', linestyle='--', label=f'最佳epoch ({best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    plt.title('训练、验证和测试准确率')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'training_metrics.png'))
    plt.savefig(os.path.join(args.save_dir, 'training_metrics.png'))  # 复制一份到根目录方便查看
    plt.close()

    log(f'所有结果已保存到 {args.save_dir} 目录')

if __name__ == "__main__":
    # Windows中的多进程支持
    multiprocessing.freeze_support()
    main() 