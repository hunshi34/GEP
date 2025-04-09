import matplotlib.pyplot as plt
import numpy as np

# 定义指标、数据集和模型
metrics = ['Accuracy', 'Precision', 'F1 Score', 'Recall']
datasets = ['Immune', 'COVID-19', 'Macrophages-easy', 'Macrophages-hard', 'Cancer']
models = ['scnym', 'scgpt', 'scbert', 'TOSICA', 'scgpt-text', 'GEP-llama1', 'GEP-llama2']

# 数据结构（从您提供的数据中提取）
data = {
    'Accuracy': {
        'Immune': [0.8792, 0.798, 0.8569, 0.8183, 0.802, 0.819, 0.893],
        'COVID-19': [0.6639, 0.794, 0.7715, 0.7376, 0.797, 0.815, 0.813],
        'Macrophages-easy': [0.6791, 0.727, 0.7017, 0.7224, 0.729, 0.726, 0.752],
        'Macrophages-hard': [0.5600, 0.630, 0.5914, 0.5865, 0.632, 0.642, 0.655],
        'Cancer': [0.9542, 0.954, 0.9528, 0.9456, 0.955, 0.959, 0.968]
    },
    'Precision': {
        'Immune': [0.8196, 0.761, 0.721, 0.725, 0.757, 0.814, 0.901],
        'COVID-19': [0.5400, 0.607, 0.5784, 0.5537, 0.584, 0.653, 0.651],
        'Macrophages-easy': [0.6014, 0.583, 0.5801, 0.5974, 0.599, 0.648, 0.695],
        'Macrophages-hard': [0.4318, 0.521, 0.4673, 0.4738, 0.502, 0.562, 0.556],
        'Cancer': [0.6615, 0.874, 0.787, 0.7831, 0.883, 0.915, 0.946]
    },
    'Recall': {
        'Immune': [0.7677, 0.749, 0.6967, 0.7241, 0.751, 0.731, 0.818],
        'COVID-19': [0.4499, 0.576, 0.4778, 0.5504, 0.555, 0.582, 0.572],
        'Macrophages-easy': [0.5279, 0.614, 0.4857, 0.6234, 0.628, 0.589, 0.610],
        'Macrophages-hard': [0.3552, 0.513, 0.3873, 0.4917, 0.498, 0.513, 0.510],
        'Cancer': [0.6276, 0.821, 0.6403, 0.8263, 0.844, 0.767, 0.868]
    },
    'F1 Score': {
        'Immune': [0.7799, 0.746, 0.6794, 0.7184, 0.748, 0.754, 0.840],
        'COVID-19': [0.4349, 0.581, 0.5029, 0.5446, 0.563, 0.599, 0.594],
        'Macrophages-easy': [0.5426, 0.593, 0.5141, 0.6064, 0.610, 0.611, 0.638],
        'Macrophages-hard': [0.3626, 0.511, 0.4600, 0.4808, 0.497, 0.531, 0.526],
        'Cancer': [0.6356, 0.832, 0.670, 0.7979, 0.854, 0.799, 0.892]
    }
}

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()  # 展平以便迭代

# 定义柱宽和位置
bar_width = 0.1
group_width = len(models) * bar_width  # 每个数据集的宽度
pos = np.arange(len(datasets)) * (group_width + 0.2)  # 每个数据集组的中心位置

# 为每个模型指定颜色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# 为每个指标绘制子图
for i, metric in enumerate(metrics):
    ax = axes[i]
    for j, dataset in enumerate(datasets):
        for k, model in enumerate(models):
            offset = pos[j] + (k - len(models) / 2) * bar_width  # 计算每个模型的偏移量
            value = data[metric][dataset][k]
            ax.bar(offset, value, width=bar_width, label=model if i == 0 and j == 0 else "", color=colors[k])

    # 设置子图样式
    ax.set_title(metric, fontsize=14, fontweight='bold')
    ax.set_xticks(pos)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylim(0.3, 1)  # Y 轴从 0.4 开始
    ax.set_ylabel(metric, fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# 添加共享图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(models), fontsize=12, bbox_to_anchor=(0.5, -0.05))

# 调整布局
plt.tight_layout(rect=[0, 0.05, 1, 1])  # 为图例留出空间

# 保存为 PDF
plt.savefig('metrics_by_dataset.pdf', format='pdf', bbox_inches='tight')

# 显示图表
plt.show()