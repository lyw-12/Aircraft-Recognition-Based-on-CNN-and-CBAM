import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 解析命令行参数
parser = argparse.ArgumentParser(description='使用训练好的模型识别飞行器图片')
parser.add_argument('--image_path', required=True, type=str, help='要识别的图片路径')
parser.add_argument('--model_path', default='models3/best_model.pth', type=str, help='模型文件路径')
parser.add_argument('--model_type', default=3, type=int, help='模型类型: 1 - 基础模型, 2 - ResNet, 3 - 注意力网络')
args = parser.parse_args()

# 类别名称
class_names = ["civil_airliner", "helicopter", "military_aircraft", "uav"]
chinese_names = ["民用客机", "直升机", "军用飞机", "无人机"]

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((140, 140)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义基础CNN模型
class BasicAircraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BasicAircraftCNN, self).__init__()

        # 第一个卷积块 - 输入: 3x140x140, 输出: 32x70x70
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二个卷积块 - 输入: 32x70x70, 输出: 64x35x35
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三个卷积块 - 输入: 64x35x35, 输出: 128x17x17
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        # 分类器部分 - 较简单的全连接层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# ResNet基本残差块
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

# 通道注意力机制
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

        # 残差连接
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

# 简化的注意力CNN模型
class SimplifiedAircraftCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimplifiedAircraftCNN, self).__init__()

        # 初始卷积层
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        # 第一个残差块组
        self.layer1 = nn.Sequential(
            SimpleResidualBlock(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二个残差块组
        self.layer2 = nn.Sequential(
            SimpleResidualBlock(64, 128, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三个残差块组
        self.layer3 = nn.Sequential(
            SimpleResidualBlock(128, 256, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 通道注意力机制
        self.attention = ChannelAttention(256)

        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 特征聚合
        self.feature_aggregation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attention(x)
        x = self.adaptive_pool(x)
        x = self.feature_aggregation(x)
        return x

def predict_image(image_path, model, transform):
    # 加载并预处理图像
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法加载图像: {e}")
        return None
    
    # 显示原始图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("原始图像")
    plt.axis('off')
    
    # 预处理图像并进行预测
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted_idx = torch.max(outputs, 1)
        prediction = predicted_idx.item()
    
    # 获取所有类别的概率
    probs = probabilities.cpu().numpy()
    
    # 显示预测结果
    plt.subplot(1, 2, 2)
    bars = plt.bar(range(len(probs)), probs)
    plt.xticks(range(len(probs)), [f"{cn}\n{chn}" for cn, chn in zip(class_names, chinese_names)], rotation=0)
    plt.title("预测概率")
    plt.xlabel("类别")
    plt.ylabel("概率")
    
    # 为最高概率添加标签
    highest_idx = np.argmax(probs)
    bars[highest_idx].set_color('r')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{probs[i]:.2f}', ha='center', va='bottom')
    
    # 显示预测结果
    title = f"预测类别: {class_names[prediction]} ({chinese_names[prediction]}), 概率: {probs[prediction]:.2f}"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存结果图像
    result_path = f"prediction_result_{os.path.basename(image_path)}.png"
    plt.savefig(result_path)
    print(f"结果已保存到: {result_path}")
    
    # 展示图像
    plt.show()
    
    return {
        'class_index': prediction,
        'class_name': class_names[prediction],
        'chinese_name': chinese_names[prediction],
        'probability': probs[prediction],
        'all_probabilities': probs
    }

def main():
    # 根据模型类型加载相应的模型
    if args.model_type == 1:
        print("加载基础CNN模型...")
        model = BasicAircraftCNN(num_classes=4)
    elif args.model_type == 2:
        print("加载ResNet模型...")
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)
    else:
        print("加载轻量级注意力网络模型...")
        model = SimplifiedAircraftCNN(num_classes=4)

    # 加载模型权重
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载模型: {args.model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 设置为评估模式
    model.to(device)
    model.eval()

    # 预测图像
    result = predict_image(args.image_path, model, transform)
    
    if result:
        print(f"\n预测结果:")
        print(f"类别: {result['class_name']} ({result['chinese_name']})")
        print(f"概率: {result['probability']:.4f}")
        print("\n所有类别概率:")
        for i, (cn, chn, prob) in enumerate(zip(class_names, chinese_names, result['all_probabilities'])):
            print(f"{cn} ({chn}): {prob:.4f}")

if __name__ == "__main__":
    main() 