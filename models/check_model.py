import torch
import os

# 设置模型路径
model_path = r"C:\Users\32501\PycharmProjects\deeplearning\models\best_model.pth"

# 检查文件是否存在
if not os.path.exists(model_path):
    print(f"模型文件不存在: {model_path}")
    exit()

# 加载模型
try:
    model_data = torch.load(model_path)
    print(f"模型文件大小: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    print(f"模型类型: {type(model_data)}")

    # 检查是否是状态字典
    if isinstance(model_data, dict):
        print("\n模型是状态字典，包含以下键:")
        for key in model_data.keys():
            print(f"- {key}")

        # 如果有模型权重，显示架构信息
        if 'state_dict' in model_data:
            print("\n模型状态字典包含以下层:")
            for key in model_data['state_dict'].keys():
                print(f"- {key}")

        # 如果保存了epoch、准确率等信息
        if 'epoch' in model_data:
            print(f"\n模型训练轮次: {model_data['epoch']}")
        if 'best_acc' in model_data:
            print(f"最佳准确率: {model_data['best_acc']:.2f}%")
    else:
        # 如果是整个模型对象
        print("\n模型结构:")
        print(model_data)

except Exception as e:
    print(f"加载模型时出错: {e}")