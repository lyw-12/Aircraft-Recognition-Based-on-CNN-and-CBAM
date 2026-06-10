import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as PathEffects

# 设置中文支持，确保您的系统支持SimHei字体，或替换为其他可用中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
except Exception as e:
    print(f"设置中文字体SimHei失败: {e}. 图像中的中文可能无法正确显示。请确保已安装SimHei字体或尝试其他中文字体。")


def add_diagram_block(ax, center_x, center_y, width, height, label, color, text_details="", detail_fontsize=7.5,
                      label_fontsize=9):
    """绘制一个表示网络层的方块，并添加标签和详细参数"""
    rect = Rectangle((center_x - width / 2, center_y - height / 2), width, height,
                     facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.9, zorder=2)
    ax.add_patch(rect)

    # 主要标签
    main_text = ax.text(center_x, center_y, label, ha='center', va='center', fontsize=label_fontsize, fontweight='bold',
                        zorder=3)
    main_text.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='white')])

    # 详细参数（如果提供）
    if text_details:
        ax.text(center_x, center_y - height / 2 - 0.15, text_details,  # 调整y坐标以在方块下方显示细节
                ha='center', va='top', fontsize=detail_fontsize, color='#333333', linespacing=1.3, zorder=3)

    # 返回块的中心点和尺寸，用于箭头连接
    return (center_x, center_y), (width, height)


def add_diagram_arrow(ax, start_pos, end_pos, connectionstyle='arc3,rad=0'):
    """绘制连接两个点的箭头"""
    arrow = FancyArrowPatch(start_pos, end_pos,
                          arrowstyle='-|>,head_length=6,head_width=4', 
                          mutation_scale=15, 
                          linewidth=1.5, color='#4A4A4A', zorder=1,
                          connectionstyle=connectionstyle)
    ax.add_patch(arrow)


def generate_basic_cnn_diagram():
    fig, ax = plt.subplots(figsize=(15, 10))  # 更大的画布高度以适应多行布局
    ax.set_xlim(0, 14)  # 减小X轴范围，使图更紧凑
    ax.set_ylim(0, 10)  # 增加Y轴范围，为多行布局提供空间
    ax.axis('off')  # 不显示坐标轴
    fig.suptitle('基础CNN模型网络结构图 (建议图片3.1)', fontsize=15, fontweight='bold', y=0.98)

    # 定义颜色主题
    colors = {
        'input': '#B0E0E6',     # PowderBlue (淡蓝色)
        'conv': '#FFDAB9',      # PeachPuff (桃色)
        'relu': '#FFFACD',      # LemonChiffon (柠檬绸色)
        'pool': '#98FB98',      # PaleGreen (苍绿色)
        'flatten': '#D8BFD8',   # Thistle (蓟色)
        'fc': '#FFC0CB',        # Pink (粉红色)
        'dropout': '#E6E6FA',   # Lavender (淡紫色)
        'output': '#FFA07A'     # LightSalmon (浅鲑色)
    }

    # 统一的块尺寸和字体大小
    block_width = 2.0  # 增加宽度
    block_height = 1.0  # 增加高度
    relu_width = block_width * 0.7
    relu_height = block_height * 0.8
    detail_fontsize = 8  # 增大字体
    label_fontsize = 9.5 # 增大字体

    # 存储每个块的信息，用于箭头连接
    blocks_info = []
    
    # 添加一个块并存储其信息
    def add_block(x, y, label, color, details, width=block_width, height=block_height, l_fontsize=label_fontsize, d_fontsize=detail_fontsize):
        pos, dims = add_diagram_block(ax, x, y, width, height, label, color, details, d_fontsize, l_fontsize)
        blocks_info.append({
            'pos': pos,
            'dims': dims,
            'label': label
        })
        return len(blocks_info) - 1  # 返回块的索引
    
    # ---- 第一行 ----
    row1_y = 8  # 第一行的y坐标
    
    # 1. 输入层
    input_details = "例如: (H, W, 3)"  # H=高度, W=宽度
    input_idx = add_block(2, row1_y, "输入层", colors['input'], input_details, width=block_width*1.1, height=block_height*1.1)
    
    # Conv1
    conv1_details = "卷积核: K1 (例如32)\n尺寸: FxF (例如3x3)\n步长: S (例如1), 填充: P (例如1)"
    conv1_idx = add_block(5, row1_y, "Conv1", colors['conv'], conv1_details)
    
    # ReLU1
    relu1_idx = add_block(8, row1_y, "ReLU1", colors['relu'], "", width=relu_width, height=relu_height, l_fontsize=8)
    
    # Pool1
    pool1_details = "类型: 最大池化\n核尺寸: F_p x F_p (例如2x2)\n步长: S_p (例如2)"
    pool1_idx = add_block(11, row1_y, "Pool1", colors['pool'], pool1_details)

    # ---- 第二行 ----
    row2_y = 5  # 第二行的y坐标
    
    # Pool2
    pool2_details = "类型: 最大池化\n核尺寸: F_p x F_p (例如2x2)\n步长: S_p (例如2)"
    pool2_idx = add_block(2, row2_y, "Pool2", colors['pool'], pool2_details)
    
    # ReLU2
    relu2_idx = add_block(5, row2_y, "ReLU2", colors['relu'], "", width=relu_width, height=relu_height, l_fontsize=8)
    
    # Conv2
    conv2_details = "卷积核: K2 (例如64)\n尺寸: FxF (例如3x3)\n步长: S (例如1), 填充: P (例如1)"
    conv2_idx = add_block(8, row2_y, "Conv2", colors['conv'], conv2_details)

    # Flatten
    flatten_idx = add_block(11, row2_y, "展平", colors['flatten'], "Flatten", width=block_width*0.9)

    # ---- 第三行 ----
    row3_y = 2  # 第三行的y坐标
    
    # FC1
    fc1_details = "神经元: N1 (例如128)"
    fc1_idx = add_block(2, row3_y, "FC1", colors['fc'], fc1_details)
    
    # ReLU3
    relu3_idx = add_block(5, row3_y, "ReLU3", colors['relu'], "", width=relu_width, height=relu_height, l_fontsize=8)
    
    # Dropout
    dropout_details = "比率: p (例如0.5)"
    dropout_idx = add_block(8, row3_y, "Dropout", colors['dropout'], dropout_details, width=block_width*0.8)
    
    # FC2 (Output)
    fc2_details = "神经元: N_classes (类别数)"
    fc2_idx = add_block(11, row3_y, "FC2 (输出)", colors['output'], fc2_details)

    # ---- 绘制箭头连接 ----
    # 第一行连接
    for i in range(input_idx, pool1_idx):
        start = blocks_info[i]['pos']
        start_dims = blocks_info[i]['dims']
        end = blocks_info[i+1]['pos']
        end_dims = blocks_info[i+1]['dims']
        
        # 水平连接，从右边到左边
        start_x = start[0] + start_dims[0]/2
        start_y = start[1]
        end_x = end[0] - end_dims[0]/2
        end_y = end[1]
        
        add_diagram_arrow(ax, (start_x, start_y), (end_x, end_y))
    
    # Pool1 到 Pool2 (第一行到第二行，需要转弯)
    pool1_pos = blocks_info[pool1_idx]['pos']
    pool1_dims = blocks_info[pool1_idx]['dims']
    pool2_pos = blocks_info[pool2_idx]['pos']
    pool2_dims = blocks_info[pool2_idx]['dims']
    
    # 从Pool1的底部到Pool2的顶部
    start_x = pool1_pos[0]
    start_y = pool1_pos[1] - pool1_dims[1]/2
    end_x = pool2_pos[0]
    end_y = pool2_pos[1] + pool2_dims[1]/2
    
    # 使用弧形连接
    add_diagram_arrow(ax, (start_x, start_y), (end_x, end_y), 
                      connectionstyle='arc3,rad=-0.5')  # 负值为顺时针弧
    
    # 第二行从左到右连接
    for i in range(pool2_idx, flatten_idx):
        start = blocks_info[i]['pos']
        start_dims = blocks_info[i]['dims']
        end = blocks_info[i+1]['pos']
        end_dims = blocks_info[i+1]['dims']
        
        # 水平连接，从右边到左边
        start_x = start[0] + start_dims[0]/2
        start_y = start[1]
        end_x = end[0] - end_dims[0]/2
        end_y = end[1]
        
        add_diagram_arrow(ax, (start_x, start_y), (end_x, end_y))
    
    # Flatten 到 FC1 (第二行到第三行，需要转弯)
    flatten_pos = blocks_info[flatten_idx]['pos']
    flatten_dims = blocks_info[flatten_idx]['dims']
    fc1_pos = blocks_info[fc1_idx]['pos']
    fc1_dims = blocks_info[fc1_idx]['dims']
    
    # 从Flatten的底部到FC1的顶部
    start_x = flatten_pos[0]
    start_y = flatten_pos[1] - flatten_dims[1]/2
    end_x = fc1_pos[0]
    end_y = fc1_pos[1] + fc1_dims[1]/2
    
    # 使用弧形连接
    add_diagram_arrow(ax, (start_x, start_y), (end_x, end_y), 
                      connectionstyle='arc3,rad=-0.5')  # 负值为顺时针弧
    
    # 第三行连接
    for i in range(fc1_idx, fc2_idx):
        start = blocks_info[i]['pos']
        start_dims = blocks_info[i]['dims']
        end = blocks_info[i+1]['pos']
        end_dims = blocks_info[i+1]['dims']
        
        # 水平连接，从右边到左边
        start_x = start[0] + start_dims[0]/2
        start_y = start[1]
        end_x = end[0] - end_dims[0]/2
        end_y = end[1]
        
        add_diagram_arrow(ax, (start_x, start_y), (end_x, end_y))
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)
    plt.savefig('basic_cnn_structure.png', dpi=300, bbox_inches='tight')
    print("基础CNN模型结构图已保存为 'basic_cnn_structure.png'")
    plt.show()  # 可选：直接显示图片


if __name__ == '__main__':
    generate_basic_cnn_diagram()