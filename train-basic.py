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
parser = argparse.ArgumentParser(description='飞行器图像分类基础CNN')
parser.add_argument('--batch_size', default=32, type=int, help='批处理大小')  # 较小的批量
parser.add_argument('--epochs', default=30, type=int, help='训练轮数')  # 减少训练轮数
parser.add_argument('--lr', default=5e-4, type=float, help='学习率')  # 较低的学习率
parser.add_argument('--data_dir', default='aircraft_dataset', type=str, help='数据集路径')
parser.add_argument('--save_dir', default='model-basic', type=str, help='模型保存路径')
parser.add_argument('--workers', default=4, type=int, help='数据加载器的线程数')
parser.add_argument('--gpu', default=0, type=int, help='指定使用的GPU编号')
args = parser.parse_args()

# 创建保存目录
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

# 设置设备
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# 数据预处理和增强
# 基础模型使用简单的数据增强
train_transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

# 创建数据加载器
train_dataset = AircraftDataset(args.data_dir, mode='train', transform=train_transform)
test_dataset = AircraftDataset(args.data_dir, mode='test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# 获取类别名称
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"类别: {class_names}")
print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# 定义基础CNN模型
# 简化的模型结构，没有使用复杂的技术如BatchNorm等
class BasicAircraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BasicAircraftCNN, self).__init__()
        # 第一个卷积块
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积块
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接分类器
        self.fc1 = nn.Linear(128 * 17 * 17, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # 卷积块2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # 卷积块3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 分类器
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 创建模型
model = BasicAircraftCNN(num_classes=num_classes)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
# 采用简单的学习率调度策略
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
        if (batch_idx + 1) % 10 == 0 or batch_idx + 1 == len(train_loader):
            print(f'Epoch: {epoch + 1} | Batch: {batch_idx + 1}/{len(train_loader)} | Loss: {running_loss / (batch_idx + 1):.4f} | Acc: {100. * correct / total:.2f}% | Time: {time.time() - start_time:.2f}s')
    
    return running_loss / len(train_loader), 100. * correct / total

# 测试函数
def test(epoch, is_final=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_predicted = []
    all_targets = []
    
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
            all_predicted.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 计算准确率
    acc = 100. * correct / total
    print(f'\nTest Epoch: {epoch + 1} | Loss: {test_loss / len(test_loader):.4f} | Acc: {acc:.2f}%\n')
    
    # 如果是最终测试或准确率更好，则保存详细指标
    if is_final or acc > best_acc:
        # 生成混淆矩阵
        cm = confusion_matrix(all_targets, all_predicted)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title(f'Epoch {epoch + 1} 混淆矩阵')
        plt.tight_layout()
        
        # 保存混淆矩阵图
        if is_final:
            plt.savefig(f'{args.save_dir}/final_confusion_matrix.png')
        else:
            plt.savefig(f'{args.save_dir}/confusion_matrix_epoch_{epoch + 1}.png')
        plt.close()
        
        # 生成分类报告
        report = classification_report(all_targets, all_predicted, target_names=class_names, digits=4)
        print(report)
        
        # 保存分类报告
        if is_final:
            with open(f'{args.save_dir}/classification_report.txt', 'w') as f:
                f.write(report)
        else:
            with open(f'{args.save_dir}/classification_report_epoch_{epoch + 1}.txt', 'w') as f:
                f.write(report)
    
    return test_loss / len(test_loader), acc

# 记录训练历史
train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0

# 训练模型
for epoch in range(args.epochs):
    # 训练一个epoch
    train_loss, train_acc = train(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # 测试一个epoch
    test_loss, test_acc = test(epoch)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    # 学习率调整
    scheduler.step(test_acc)
    
    # 保存最佳模型
    if test_acc > best_acc:
        print(f'准确率提升: {best_acc:.2f}% -> {test_acc:.2f}%')
        best_acc = test_acc
        torch.save(model.state_dict(), f'{args.save_dir}/best_model.pth')
    
    # 定期保存检查点
    if (epoch + 1) % 5 == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc
        }, f'{args.save_dir}/checkpoint_epoch_{epoch + 1}.pth')

# 最终测试
print('加载最佳模型进行最终测试...')
model.load_state_dict(torch.load(f'{args.save_dir}/best_model.pth'))
final_loss, final_acc = test(args.epochs - 1, is_final=True)
print(f'最终测试准确率: {final_acc:.2f}%')

# 绘制训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='训练损失')
plt.plot(test_losses, label='测试损失')
plt.xlabel('Epoch')
plt.ylabel('损失')
plt.legend()
plt.title('训练和测试损失')

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='训练准确率')
plt.plot(test_accs, label='测试准确率')
plt.xlabel('Epoch')
plt.ylabel('准确率 (%)')
plt.legend()
plt.title('训练和测试准确率')

plt.tight_layout()
plt.savefig(f'{args.save_dir}/training_metrics.png')
plt.close()

# 保存训练历史
history = {
    'train_loss': train_losses,
    'train_acc': train_accs,
    'test_loss': test_losses,
    'test_acc': test_accs,
    'best_acc': best_acc
}
np.save(f'{args.save_dir}/training_history.npy', history)

print(f'训练完成! 最佳测试准确率: {best_acc:.2f}%')

# 保存模型架构信息
model_info = {
    'model_name': 'BasicAircraftCNN',
    'num_params': sum(p.numel() for p in model.parameters()),
    'input_size': (3, 140, 140),
    'output_size': num_classes
}

with open(f'{args.save_dir}/model_info.txt', 'w') as f:
    for key, value in model_info.items():
        f.write(f'{key}: {value}\n')

print(f'模型信息已保存到 {args.save_dir}/model_info.txt')


