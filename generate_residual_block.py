import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, ConnectionPatch
import matplotlib.patheffects as PathEffects

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def add_block(ax, position, width, height, color, text, alpha=0.7):
    rect = Rectangle(position, width, height, linewidth=1, edgecolor='black', facecolor=color, alpha=alpha)
    ax.add_patch(rect)
    
    # 添加文本
    if text:
        tx = position[0] + width/2
        ty = position[1] + height/2
        t = ax.text(tx, ty, text, ha='center', va='center', fontsize=10, fontweight='bold')
        t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
    
    return rect

def add_arrow(ax, start_point, end_point, connectionstyle='arc3,rad=-0.0'):
    arrow = ConnectionPatch(start_point, end_point, 'data', 'data',
                          arrowstyle='->', 
                          connectionstyle=connectionstyle,
                          mutation_scale=15, 
                          linewidth=1,
                          color='black')
    ax.add_patch(arrow)
    return arrow

def generate_residual_block():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')
    
    # 定义颜色
    colors = {
        'input': '#AEDFF7',   # 浅蓝色
        'conv': '#FFA07A',    # 浅红色
        'bn': '#98FB98',      # 浅绿色
        'relu': '#FFD700',    # 黄色
        'shortcut': '#DDA0DD' # 紫色
    }
    
    # 设置标题
    fig.suptitle('ResidualBlock详细结构图', fontsize=18, fontweight='bold')
    ax1.set_title('情况1: 直接恒等映射 (Identity Shortcut)', fontsize=14)
    ax2.set_title('情况2: 使用1×1卷积进行下采样/维度匹配', fontsize=14)
    
    # 设置坐标范围
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
    
    # ===== 第一部分: 直接恒等映射 =====
    # 绘制输入
    input_block = add_block(ax1, (2, 9), 6, 0.7, colors['input'], "输入: x")
    
    # 绘制主路径
    conv1 = add_block(ax1, (3, 7.5), 4, 0.7, colors['conv'], "Conv2D (3×3)")
    bn1 = add_block(ax1, (3, 6.5), 4, 0.7, colors['bn'], "BatchNormalization")
    relu1 = add_block(ax1, (3, 5.5), 4, 0.7, colors['relu'], "ReLU")
    
    conv2 = add_block(ax1, (3, 4.5), 4, 0.7, colors['conv'], "Conv2D (3×3)")
    bn2 = add_block(ax1, (3, 3.5), 4, 0.7, colors['bn'], "BatchNormalization")
    
    # 添加主路径箭头
    add_arrow(ax1, (5, 9), (5, 7.5+0.7))
    add_arrow(ax1, (5, 7.5), (5, 6.5+0.7))
    add_arrow(ax1, (5, 6.5), (5, 5.5+0.7))
    add_arrow(ax1, (5, 5.5), (5, 4.5+0.7))
    add_arrow(ax1, (5, 4.5), (5, 3.5+0.7))
    
    # 绘制快捷连接
    shortcut_start = (8, 9)
    shortcut_end = (8, 2.5+0.7)
    ax1.plot([shortcut_start[0], shortcut_start[0]], [shortcut_start[1], shortcut_end[1]], 
             'k--', linewidth=2, color='blue')
    ax1.text(8.3, 5.5, "恒等映射", fontsize=10, rotation=90, va='center', ha='center')
    
    # 绘制加法和输出
    addition_block = add_block(ax1, (4, 2.5), 2, 0.7, 'white', "+")
    ax1.text(5, 2, "F(x) + x", fontsize=10, ha='center')
    
    # 向加法块添加箭头
    add_arrow(ax1, (5, 3.5), (5, 2.5+0.7))  # 从BN到加号
    add_arrow(ax1, (8, 2.5+0.35), (6, 2.5+0.35))  # 从快捷连接到加号
    
    # 输出和ReLU
    output_relu = add_block(ax1, (3, 1.5), 4, 0.7, colors['relu'], "ReLU")
    add_arrow(ax1, (5, 2.5), (5, 1.5+0.7))
    
    output = add_block(ax1, (3, 0.5), 4, 0.7, colors['input'], "输出")
    add_arrow(ax1, (5, 1.5), (5, 0.5+0.7))
    
    # ===== 第二部分: 1×1卷积用于下采样 =====
    # 绘制输入
    input_block = add_block(ax2, (2, 9), 6, 0.7, colors['input'], "输入: x")
    
    # 绘制主路径
    conv1 = add_block(ax2, (3, 7.5), 4, 0.7, colors['conv'], "Conv2D (3×3, stride=2)")
    bn1 = add_block(ax2, (3, 6.5), 4, 0.7, colors['bn'], "BatchNormalization")
    relu1 = add_block(ax2, (3, 5.5), 4, 0.7, colors['relu'], "ReLU")
    
    conv2 = add_block(ax2, (3, 4.5), 4, 0.7, colors['conv'], "Conv2D (3×3)")
    bn2 = add_block(ax2, (3, 3.5), 4, 0.7, colors['bn'], "BatchNormalization")
    
    # 添加主路径箭头
    add_arrow(ax2, (5, 9), (5, 7.5+0.7))
    add_arrow(ax2, (5, 7.5), (5, 6.5+0.7))
    add_arrow(ax2, (5, 6.5), (5, 5.5+0.7))
    add_arrow(ax2, (5, 5.5), (5, 4.5+0.7))
    add_arrow(ax2, (5, 4.5), (5, 3.5+0.7))
    
    # 绘制快捷连接 (带有1×1卷积)
    shortcut_conv = add_block(ax2, (7, 7.5), 2, 0.7, colors['conv'], "Conv2D (1×1, stride=2)")
    shortcut_bn = add_block(ax2, (7, 6.5), 2, 0.7, colors['bn'], "BN")
    
    # 添加快捷路径箭头
    add_arrow(ax2, (8, 9), (8, 7.5+0.7))
    add_arrow(ax2, (8, 7.5), (8, 6.5+0.7))
    add_arrow(ax2, (8, 6.5), (8, 2.5+0.7))
    ax2.text(8.3, 4.5, "参数化快捷连接", fontsize=10, rotation=90, va='center', ha='center')
    
    # 绘制加法和输出
    addition_block = add_block(ax2, (4, 2.5), 2, 0.7, 'white', "+")
    ax2.text(5, 2, "F(x) + W·x", fontsize=10, ha='center')
    
    # 向加法块添加箭头
    add_arrow(ax2, (5, 3.5), (5, 2.5+0.7))  # 从BN到加号
    add_arrow(ax2, (8, 2.5+0.35), (6, 2.5+0.35))  # 从快捷连接到加号
    
    # 输出和ReLU
    output_relu = add_block(ax2, (3, 1.5), 4, 0.7, colors['relu'], "ReLU")
    add_arrow(ax2, (5, 2.5), (5, 1.5+0.7))
    
    output = add_block(ax2, (3, 0.5), 4, 0.7, colors['input'], "输出")
    add_arrow(ax2, (5, 1.5), (5, 0.5+0.7))
    
    # 添加图例注释
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    ax1.text(0.5, 0.95, '示例：相同维度，无需下采样', transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    ax2.text(0.5, 0.95, '示例：需下采样或通道数变化', transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('residual_block.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("ResidualBlock结构图已保存为'residual_block.png'")

if __name__ == '__main__':
    generate_residual_block() 