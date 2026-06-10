import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import glob
import shutil
from PIL import Image, ImageDraw, ImageFont

def setup_chinese_font():
    """配置matplotlib中文字体"""
    print("配置matplotlib中文字体...")
    
    # 检查Windows常见中文字体
    windows_fonts = [
        'C:/Windows/Fonts/simhei.ttf',  # 黑体
        'C:/Windows/Fonts/msyh.ttc',    # 微软雅黑
        'C:/Windows/Fonts/simsun.ttc',  # 宋体
    ]
    
    font_found = False
    for font_path in windows_fonts:
        if os.path.exists(font_path):
            font_name = os.path.basename(font_path).split('.')[0]
            print(f"找到系统字体: {font_name}")
            
            # 配置matplotlib字体
            plt.rcParams['font.sans-serif'] = [font_name, 'SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            
            font_found = True
            break
    
    if not font_found:
        print("未找到系统中文字体")
    
    return font_found

def fix_image_with_pil(img_path):
    """使用PIL直接在图片上添加中文文本覆盖方块字"""
    try:
        # 打开图片
        img = Image.open(img_path)
        draw = ImageDraw.Draw(img)
        
        # 尝试加载系统中文字体
        font_path = None
        for path in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simsun.ttc']:
            if os.path.exists(path):
                font_path = path
                break
        
        if font_path is None:
            print(f"未找到系统中文字体，无法处理图片: {img_path}")
            return False
        
        # 创建字体对象
        title_font = ImageFont.truetype(font_path, 20)
        label_font = ImageFont.truetype(font_path, 16)
        
        # 根据图片类型添加文本
        if 'confusion_matrix' in img_path:
            # 如果是混淆矩阵图，添加标题和轴标签
            if 'test_' in os.path.basename(img_path):
                prefix = '测试集'
                epoch = os.path.basename(img_path).split('epoch_')[1].split('.')[0]
            elif 'val_' in os.path.basename(img_path):
                prefix = '验证集'
                epoch = os.path.basename(img_path).split('epoch_')[1].split('.')[0]
            else:
                prefix = '测试集'
                epoch = 'final'
            
            # 在图片左上角添加白色背景，覆盖原有方块字
            width, height = img.size
            
            # 绘制标题背景和文本
            draw.rectangle([(width/2-150, 20), (width/2+150, 50)], fill=(255, 255, 255))
            draw.text((width/2-120, 25), f"{prefix} Epoch {epoch} 混淆矩阵", fill=(0, 0, 0), font=title_font)
            
            # 绘制x轴和y轴标签背景和文本
            draw.rectangle([(width/2-50, height-40), (width/2+50, height-10)], fill=(255, 255, 255))
            draw.text((width/2-30, height-35), "预测标签", fill=(0, 0, 0), font=label_font)
            
            draw.rectangle([(20, height/2-50), (60, height/2+50)], fill=(255, 255, 255))
            draw.text((25, height/2), "真实标签", fill=(0, 0, 0), font=label_font)
            
        elif 'training_metrics' in img_path:
            # 如果是训练指标图，添加标题和轴标签
            width, height = img.size
            
            # 上半部分(损失曲线)
            draw.rectangle([(width/2-100, 20), (width/2+100, 50)], fill=(255, 255, 255))
            draw.text((width/2-90, 25), "训练、验证和测试损失", fill=(0, 0, 0), font=title_font)
            
            draw.rectangle([(width/2-50, height/2-30), (width/2+50, height/2-10)], fill=(255, 255, 255))
            draw.text((width/2-20, height/2-25), "损失", fill=(0, 0, 0), font=label_font)
            
            # 下半部分(准确率曲线)
            draw.rectangle([(width/2-100, height/2+20), (width/2+100, height/2+50)], fill=(255, 255, 255))
            draw.text((width/2-90, height/2+25), "训练、验证和测试准确率", fill=(0, 0, 0), font=title_font)
            
            draw.rectangle([(width/2-50, height-30), (width/2+50, height-10)], fill=(255, 255, 255))
            draw.text((width/2-40, height-25), "准确率 (%)", fill=(0, 0, 0), font=label_font)
        
        # 保存修改后的图片
        img.save(img_path)
        print(f"成功修复图片: {img_path}")
        return True
    
    except Exception as e:
        print(f"处理图片 {img_path} 时出错: {str(e)}")
        return False

def main():
    print("=" * 50)
    print("中文字体问题快速修复工具")
    print("=" * 50)
    print("此脚本将修复由newtrain2.py生成的图片中中文显示为方块的问题。")
    print("通过在现有图片上直接覆盖中文文本，不需要重新运行模型。")
    print("=" * 50)
    
    save_dir = input("请输入模型保存目录 (默认为 newmodels-resnet): ") or 'newmodels-resnet'
    plots_dir = os.path.join(save_dir, 'plots')
    
    if not os.path.exists(plots_dir):
        print(f"错误: 无法找到图片目录 {plots_dir}")
        return
    
    # 获取所有混淆矩阵图片和训练指标图片
    cm_images = glob.glob(os.path.join(plots_dir, '*confusion_matrix*.png'))
    metrics_images = glob.glob(os.path.join(plots_dir, 'training_metrics.png'))
    metrics_images += glob.glob(os.path.join(save_dir, 'training_metrics.png'))
    
    all_images = cm_images + metrics_images
    
    if not all_images:
        print("未找到需要修复的图片")
        return
    
    print(f"找到 {len(all_images)} 个图片需要修复")
    
    # 备份原始图片
    backup_dir = os.path.join(save_dir, 'plots_backup')
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
        print(f"创建备份目录: {backup_dir}")
        
        # 只在第一次运行时备份
        for img_path in all_images:
            img_name = os.path.basename(img_path)
            backup_path = os.path.join(backup_dir, img_name)
            shutil.copy2(img_path, backup_path)
            print(f"已备份: {img_path} -> {backup_path}")
    
    # 修复图片
    fixed_count = 0
    for img_path in all_images:
        if fix_image_with_pil(img_path):
            fixed_count += 1
    
    print(f"修复完成! 成功修复 {fixed_count}/{len(all_images)} 个图片")
    if fixed_count < len(all_images):
        print(f"原始图片备份在: {backup_dir}")

if __name__ == "__main__":
    main() 