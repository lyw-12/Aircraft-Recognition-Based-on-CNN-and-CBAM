import pandas as pd
from tabulate import tabulate

# 创建模型对比数据
data = {
    "模型名称": ["基础CNN", "ResNet-CBAM"],
    "测试准确率(%)": [92.26, 97.59],
    "参数量(M)": [5.02, 2.82],  # 5,023,332 -> 5.02M, 2,824,486 -> 2.82M
    "总训练时间(分钟)": [9.04, 52.71],  # 542.11秒 -> 9.04分钟, 3162.88秒 -> 52.71分钟
    "宏平均F1-Score": [0.9227, 0.9759]  # 从分类报告中获取
}

# 创建DataFrame
df = pd.DataFrame(data)

# 使用tabulate格式化表格
table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)

# 输出表格
print(table)

# 保存为CSV
df.to_csv("model_comparison.csv", index=False)
print("表格已保存为model_comparison.csv")