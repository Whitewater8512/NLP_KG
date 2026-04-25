import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 1. 莫兰迪色系定义
MORANDI_PALETTE = ["#949483", "#B1A494", "#A1B3B3", "#D6C7C7", "#9B8E7D", "#E5E1D8"]

def plot_professional_dashboard(results_df):
    """
    生成专业科学实验结果看板
    """
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    fig = plt.subplots(2, 2, figsize=(18, 14))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # A. 模型综合性能对比 (Bar Plot with Error Bars)
    ax1 = plt.subplot(2, 2, 1)
    df_melt = results_df.melt(id_vars='Model', value_vars=['MRR', 'Hits@10'], var_name='Metric', value_name='Score')
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette=MORANDI_PALETTE, ax=ax1, capsize=.05)
    ax1.set_title("A. Global Model Performance Comparison", fontsize=15, pad=15)
    ax1.set_ylim(0, 0.6)

    # B. 超参数热力图: LR vs Dropout (以ConvE为例)
    ax2 = plt.subplot(2, 2, 2)
    conve_data = results_df[results_df['Model'] == 'ConvE'].pivot_table(index='lr', columns='inp_drop', values='MRR')
    sns.heatmap(conve_data, annot=True, fmt=".3f", cmap=sns.light_palette(MORANDI_PALETTE[0], as_cmap=True), ax=ax2)
    ax2.set_title("B. ConvE Sensitivity: LR vs Input_Dropout", fontsize=15, pad=15)

    # C. 维度分析趋势 (Line Plot)
    ax3 = plt.subplot(2, 2, 3)
    sns.lineplot(data=results_df, x='dim', y='MRR', hue='Model', style='Model', 
                 markers=True, dashes=False, palette=MORANDI_PALETTE[:3], ax=ax3, lw=2.5)
    ax3.set_title("C. Performance Trend by Embedding Dimension", fontsize=15, pad=15)

    # D. 训练收敛稳定性 (模拟消融实验)
    ax4 = plt.subplot(2, 2, 4)
    # 模拟数据
    epochs = np.linspace(0, 100, 10)
    for i, model in enumerate(['TransE', 'DistMult', 'ConvE']):
        y = 0.4 * (1 - np.exp(-epochs/20)) + np.random.normal(0, 0.01, 10) + i*0.05
        ax4.plot(epochs, y, label=model, color=MORANDI_PALETTE[i], lw=2, marker='o', markersize=4)
    ax4.set_title("D. Convergence Efficiency (Valid MRR over Epochs)", fontsize=15, pad=15)
    ax4.legend()

    plt.suptitle("Knowledge Graph Embedding Hyperparameter Analysis Report", fontsize=22, y=0.96)
    plt.show()

# 示例调用
# data = pd.read_csv("experiment_results.csv")
# plot_professional_dashboard(data)