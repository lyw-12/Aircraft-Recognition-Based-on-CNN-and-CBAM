import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 各层配置
blocks = [
    {"title": "初始卷积层", "desc": "Conv2d(3,64,7×7)\nBN+ReLU+MaxPool", "color": "#A2C4C9"},
    {"title": "残差层1", "desc": "输入64→输出64\n2个残差块\n步长1", "color": "#B6D7A8"},
    {"title": "残差层2", "desc": "输入64→输出128\n2个残差块\n步长2", "color": "#B6D7A8"},
    {"title": "残差层3", "desc": "输入128→输出256\n2个残差块\n步长2", "color": "#B6D7A8"},
    {"title": "CBAM注意力模块", "desc": "通道+空间注意力", "color": "#FFD966"},
    {"title": "全局平均池化", "desc": "AdaptiveAvgPool2d(1,1)", "color": "#D9D2E9"},
    {"title": "全连接分类器", "desc": "Linear(256→128)\nReLU+Dropout\nLinear(128→4)", "color": "#F6B26B"},
    {"title": "输出层", "desc": "4类", "color": "#EA9999"},
]

fig, ax = plt.subplots(figsize=(16, 4))
ax.axis('off')

x = 0.5
box_width = 2.8
box_height = 1.6
for i, block in enumerate(blocks):
    box = FancyBboxPatch((x, 1), box_width, box_height,
                        boxstyle="round,pad=0.15",
                        linewidth=2,
                        edgecolor="#555",
                        facecolor=block["color"])
    ax.add_patch(box)
    ax.text(x + box_width/2, 1 + box_height/2 + 0.25, block["title"], ha='center', va='center', fontsize=15, fontweight='bold')
    ax.text(x + box_width/2, 1 + box_height/2 - 0.35, block["desc"], ha='center', va='center', fontsize=12)
    # 箭头
    if i < len(blocks) - 1:
        ax.annotate('', xy=(x + box_width, 1 + box_height/2), xytext=(x + box_width + 0.5, 1 + box_height/2),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#555"))
    x += box_width + 0.7

ax.set_xlim(0, x + 0.5)
ax.set_ylim(0, 4)
plt.title('ResNet+CBAM改进模型网络结构图', fontsize=18, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('resnet_cbam_structure.png', dpi=200)
plt.show()

print("结构图已保存为resnet_cbam_structure.png") 