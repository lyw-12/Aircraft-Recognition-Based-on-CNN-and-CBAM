import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')

# 输入层
ax.add_patch(Rectangle((0.5, 1.5), 1, 1, fill=False, lw=2))
ax.text(1, 2, '2', fontsize=24, ha='center', va='center')
ax.text(1, 1.4, '输入层', fontsize=12, ha='center', va='top')

# 第一卷积
for i in range(5):
    ax.add_patch(Rectangle((2+i*0.1, 1.3+i*0.1), 1, 1, fill=False, lw=1.5))
ax.text(2.5, 1.2, '卷积', fontsize=12, ha='center', va='top')

# 第一池化
for i in range(4):
    ax.add_patch(Rectangle((3.5+i*0.1, 1.1+i*0.1), 0.8, 0.8, fill=False, lw=1.2))
ax.text(4, 1.0, '池化', fontsize=12, ha='center', va='top')

# 第二卷积
for i in range(5):
    ax.add_patch(Rectangle((5+i*0.1, 0.9+i*0.1), 0.7, 0.7, fill=False, lw=1.2))
ax.text(5.5, 0.8, '卷积', fontsize=12, ha='center', va='top')

# 第二池化
for i in range(3):
    ax.add_patch(Rectangle((6.5+i*0.1, 0.7+i*0.1), 0.5, 0.5, fill=False, lw=1))
ax.text(7, 0.6, '池化', fontsize=12, ha='center', va='top')

# 全连接层
ax.add_patch(Rectangle((8, 1.2), 0.2, 1.2, fill=False, lw=2))
ax.text(8.1, 1.1, '全连接', fontsize=12, ha='center', va='top', rotation=90)
ax.add_patch(Rectangle((8.5, 1.4), 0.2, 0.8, fill=False, lw=2))
ax.text(8.6, 1.3, '全连接', fontsize=12, ha='center', va='top', rotation=90)

# 输出层
ax.add_patch(Rectangle((9, 1.7), 0.2, 0.3, fill=False, lw=2))
ax.text(9.25, 1.85, '1\n2\n3', fontsize=12, ha='left', va='center')
ax.text(9.1, 1.6, '输出层', fontsize=12, ha='center', va='top', rotation=90)

# 卷积层大括号
ax.plot([1.7, 7.5], [3.1, 3.1], color='k')
ax.plot([1.7, 1.7], [3.1, 0.8], color='k')
ax.plot([7.5, 7.5], [3.1, 0.8], color='k')
ax.text(4.6, 3.25, '卷积层', fontsize=12, ha='center', va='bottom')

# 全连接层大括号（只包住两个全连接层，不包括输出层）
ax.plot([7.9, 8.9], [2.8, 2.8], color='k')
ax.plot([7.9, 7.9], [2.8, 1.1], color='k')
ax.plot([8.9, 8.9], [2.8, 1.1], color='k')
ax.text(8.4, 2.95, '全连接层', fontsize=12, ha='center', va='bottom')

# 箭头
def draw_arrow(xyA, xyB):
    arrow = FancyArrowPatch(xyA, xyB, arrowstyle='->', mutation_scale=15, lw=1.5)
    ax.add_patch(arrow)

draw_arrow((1.5, 2), (2, 2))
draw_arrow((3, 2), (3.5, 1.7))
draw_arrow((4.3, 1.7), (5, 1.3))
draw_arrow((5.9, 1.3), (6.5, 1))
draw_arrow((7.1, 1), (8, 1.8))
draw_arrow((8.2, 2), (8.5, 1.8))
draw_arrow((8.7, 1.8), (9, 1.85))

plt.xlim(0, 10)
plt.ylim(0.3, 3.5)
plt.tight_layout()
plt.show()