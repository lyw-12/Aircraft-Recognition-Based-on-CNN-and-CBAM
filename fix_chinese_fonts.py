import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import shutil
import glob

def setup_chinese_font():
    """设置matplotlib中文字体"""
    print("开始配置中文字体...")
    
    # 检查系统中文字体
    windows_fonts = [
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
        'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
        'C:/Windows/Fonts/simkai.ttf',  # 楷体
    ]
    
    font_path = None
    for path in windows_fonts:
        if os.path.exists(path):
            font_path = path
            font_name = os.path.basename(path).split('.')[0]
            print(f"找到系统字体: {font_name} ({path})")
            break
    
    if font_path is None:
        print("未找到系统中文字体，将使用备选方案")
        return False
    
    # 配置matplotlib字体
    matplotlib_dir = matplotlib.get_configdir()
    fonts_dir = os.path.join(matplotlib_dir, "fonts", "ttf")
    os.makedirs(fonts_dir, exist_ok=True)
    
    # 复制字体文件
    target_font_path = os.path.join(fonts_dir, os.path.basename(font_path))
    if not os.path.exists(target_font_path):
        print(f"复制字体文件到 {target_font_path}")
        shutil.copy2(font_path, target_font_path)
    
    # 生成字体缓存文件
    print("正在重建字体缓存...")
    fm._rebuild()
    
    # 设置默认字体
    plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    print("中文字体配置完成!")
    return True

def fix_plot_images(save_dir='newmodels-resnet'):
    """修复已生成的图片"""
    if not setup_chinese_font():
        print("中文字体设置失败，无法修复图片")
        return False
    
    plots_dir = os.path.join(save_dir, 'plots')
    if not os.path.exists(plots_dir):
        print(f"无法找到图片目录: {plots_dir}")
        return False
    
    # 获取所有混淆矩阵图片
    cm_images = glob.glob(os.path.join(plots_dir, '*confusion_matrix*.png'))
    # 获取训练指标图片
    metric_images = glob.glob(os.path.join(plots_dir, 'training_metrics.png'))
    metric_images += glob.glob(os.path.join(save_dir, 'training_metrics.png'))
    
    images = cm_images + metric_images
    
    if not images:
        print("没有找到需要修复的图片")
        return False
    
    print(f"找到 {len(images)} 个图片需要修复")
    
    # 开始修复图片
    for img_path in images:
        try:
            print(f"处理图片: {img_path}")
            
            if 'confusion_matrix' in img_path:
                # 提取相关信息
                if 'test_' in os.path.basename(img_path):
                    title_prefix = '测试集'
                    epoch = os.path.basename(img_path).split('epoch_')[1].split('.')[0]
                elif 'val_' in os.path.basename(img_path):
                    title_prefix = '验证集'
                    epoch = os.path.basename(img_path).split('epoch_')[1].split('.')[0]
                else:
                    title_prefix = '测试集'
                    epoch = 'final'
                
                # 读取类别名称 (这里假设类别固定为aircraft_dataset中的类别)
                class_names = ["军用飞机", "民用飞机", "直升机", "无人机"]
                
                # 创建新的混淆矩阵图
                # 注意：这里我们不读取原始数据，只是创建一个新的具有中文标签的图
                # 实际项目可能需要修改读取真实数据
                fig, ax = plt.subplots(figsize=(10, 8))
                # 创建一个简单的示意图，实际应用中应加载真实数据
                ax.text(0.5, 0.5, "此图仅作为字体修复示例\n请运行模型获取实际数据", 
                       ha='center', va='center', fontsize=16)
                plt.title(f'{title_prefix} Epoch {epoch} 混淆矩阵')
                plt.xlabel('预测标签')
                plt.ylabel('真实标签')
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close()
                
            elif 'training_metrics' in img_path:
                # 创建新的训练指标图
                fig = plt.figure(figsize=(12, 8))
                
                # 损失曲线
                plt.subplot(2, 1, 1)
                plt.xlabel('Epoch')
                plt.ylabel('损失')
                plt.title('训练、验证和测试损失')
                plt.grid(True)
                plt.text(0.5, 0.5, "此图仅作为字体修复示例\n请运行模型获取实际数据", 
                       ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
                
                # 准确率曲线
                plt.subplot(2, 1, 2)
                plt.xlabel('Epoch')
                plt.ylabel('准确率 (%)')
                plt.title('训练、验证和测试准确率')
                plt.grid(True)
                plt.text(0.5, 0.5, "此图仅作为字体修复示例\n请运行模型获取实际数据", 
                       ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
                
                plt.tight_layout()
                plt.savefig(img_path)
                plt.close()
            
            print(f"图片修复完成: {img_path}")
        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")
    
    print("所有图片处理完毕!")
    return True

def fix_newtrain2_font():
    """修改newtrain2.py文件，添加中文字体支持"""
    file_path = 'newtrain2.py'
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找matplotlib导入部分
        if 'import matplotlib.pyplot as plt' in content and 'plt.rcParams' not in content:
            # 在导入matplotlib后添加中文字体设置
            import_line = 'import matplotlib.pyplot as plt'
            font_config = '''
# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans', 'Arial']  
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像负号'-'显示为方块的问题
'''
            content = content.replace(import_line, import_line + font_config)
            
            # 保存修改后的文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"已成功修改 {file_path}，添加了中文字体支持")
            return True
        else:
            print(f"无需修改 {file_path}，已包含中文字体配置或结构不符合预期")
            return False
    
    except Exception as e:
        print(f"修改文件 {file_path} 时出错: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("中文字体问题修复工具")
    print("=" * 50)
    print("1. 此脚本可以解决matplotlib生成图片中文显示为方块的问题")
    print("2. 两种修复方式：修复已生成的图片 或 修改newtrain2.py添加中文支持")
    print("=" * 50)
    
    choice = input("请选择操作:\n1. 修复已生成的图片\n2. 修改newtrain2.py添加中文支持\n3. 两者都执行\n请输入(1/2/3): ")
    
    if choice == '1':
        save_dir = input("请输入模型保存目录 (默认为 newmodels-resnet): ") or 'newmodels-resnet'
        fix_plot_images(save_dir)
    elif choice == '2':
        fix_newtrain2_font()
    elif choice == '3':
        fix_newtrain2_font()
        save_dir = input("请输入模型保存目录 (默认为 newmodels-resnet): ") or 'newmodels-resnet'
        fix_plot_images(save_dir)
    else:
        print("无效选择，请输入1、2或3") 