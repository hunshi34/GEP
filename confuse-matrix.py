import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    "Set": ["FULL", "W/O T", "W/O C", "W/O G"],
    "G": ["✓", "✓", "✓", "✗"],
    "C": ["✓", "✓", "✗", "✓"],
    "T": ["✓", "✗", "✓", "✓"],
    "Acc": [0.830, 0.833, 0.813, 0.812],
    "Pre": [0.822, 0.797, 0.832, 0.772],
    "Rec": [0.752, 0.756, 0.779, 0.716],
    "F1": [0.770, 0.772, 0.796, 0.729]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 创建画布
fig, ax = plt.subplots(figsize=(6, 2.5))  # 调整画布大小以容纳描述
ax.axis('off')  # 隐藏坐标轴

# 绘制表格
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',  # 单元格内容居中
                 loc='center',
                 colColours=['#f0f0f0']*len(df.columns),  # 表头背景色
                 colWidths=[0.15, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15])  # 调整列宽

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)  # 调整表格行高

# 自定义边框样式（三线表）
for (i, j), cell in table.get_celld().items():
    if i == 0:  # 表头行
        cell.set_text_props(weight='bold')
        cell.set_linewidth(0.5)
        cell._edgecolor = ['none', 'none', 'black', 'none']  # 仅下边框（普通线）
        cell.set_height(0.1)  # 调整表头行高
    elif i == 1:  # 第一行数据（顶部加粗线）
        cell.set_linewidth(1.0)  # 加粗顶部线
        cell._edgecolor = ['black', 'none', 'black', 'none']  # 上下边框（顶部加粗）
    elif i == len(df) + 1:  # 最后一行（底部加粗线）
        cell.set_linewidth(1.0)  # 加粗底部线
        cell._edgecolor = ['black', 'none', 'none', 'none']  # 仅上边框（加粗）
    else:  # 其他数据行
        cell.set_linewidth(0.5)
        cell._edgecolor = ['none', 'none', 'none', 'none']  # 无边框（三线表无中间横线）

# 添加描述文本
description = "Table 1. Ablation Study Results for \"Cell Description Information\" on the Immune Dataset. GCD: Gene Contextual Description, CCM: Cell Characteristic Metadata, TCS: Task and Category Specification."
plt.figtext(0.1, 0.05, description, wrap=True, horizontalalignment='left', fontsize=8)

# 保存为 PDF
plt.savefig("table_with_caption.pdf", bbox_inches='tight', format='pdf')
plt.close()