import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

MORANDI_PALETTE = ["#949483", "#B1A494", "#A1B3B3", "#D6C7C7", "#9B8E7D", "#E5E1D8", "#7A8B8B"]

def plot_model_comparison(best_results_df, best_histories_dict):
    """
    绘制各模型【最佳配置】的对比看板
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    # 这里的 models 应该是经过筛选后的最佳模型 ID
    models = list(best_histories_dict.keys())
    
    # ---------------------------------------------------------
    # 图 A: 最佳配置的训练 Loss 收敛情况 (平滑处理)
    # ---------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    for i, model_id in enumerate(models):
        hist = best_histories_dict[model_id]
        ax1.plot(hist['epoch'], hist['train_loss'], label=model_id, 
                 color=MORANDI_PALETTE[i], lw=2.5, alpha=0.8)
    ax1.set_title("A. Best Config: Training Loss Convergence", fontsize=14, pad=10, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Avg Batch Loss", fontsize=12)
    ax1.legend(fontsize=9)

    # ---------------------------------------------------------
    # 图 B: 最佳配置的验证集 MRR 攀升曲线
    # ---------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    for i, model_id in enumerate(models):
        hist = best_histories_dict[model_id]
        valid_epochs = [e for e, mrr in zip(hist['epoch'], hist['valid_mrr']) if mrr is not None]
        valid_mrrs = [mrr for mrr in hist['valid_mrr'] if mrr is not None]
        ax2.plot(valid_epochs, valid_mrrs, label=model_id, 
                 color=MORANDI_PALETTE[i], marker='o', markersize=4, lw=2)
    ax2.set_title("B. Best Config: Validation MRR Progress", fontsize=14, pad=10, fontweight='bold')
    ax2.set_ylabel("Validation MRR", fontsize=12)
    ax2.legend(fontsize=9)

    # ---------------------------------------------------------
    # 图 C: 最佳配置的计算效率对比 (时间 vs 性能)
    # ---------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    efficiency_data = []
    for model_id in models:
        hist = best_histories_dict[model_id]
        avg_train_time = np.mean(hist['train_time'])
        # 关联最终的 MRR 结果
        final_mrr = next(item['Test MRR'] for item in best_results_df if item['Model_ID'] == model_id)
        efficiency_data.append({'Model': model_id, 'Train Time/Epoch (s)': avg_train_time, 'MRR': final_mrr})
    
    df_eff = pd.DataFrame(efficiency_data)
    # 使用散点图展示效率：X轴为时间，Y轴为性能，点的大小代表模型优劣
    sns.scatterplot(data=df_eff, x='Train Time/Epoch (s)', y='MRR', hue='Model', 
                    palette=MORANDI_PALETTE[:len(models)], s=200, ax=ax3)
    ax3.set_title("C. Efficiency vs. Performance (Best Runs)", fontsize=14, pad=10, fontweight='bold')

    # ---------------------------------------------------------
    # 图 D: 测试集最终指标横向对比
    # ---------------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    df_metrics = pd.DataFrame(best_results_df)
    df_melted = df_metrics.melt(id_vars='Model_ID', value_vars=['Test MRR', 'Test Hits@10'], 
                                var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model_ID', 
                palette=MORANDI_PALETTE[:len(models)], ax=ax4)
    ax4.set_title("D. Final Best-in-Class Performance", fontsize=14, pad=10, fontweight='bold')
    
    plt.suptitle("KGE Excellence: Comparison of Best Performing Configurations", 
                 fontsize=18, y=0.98, fontweight='bold', color="#4A4A4A")
    plt.savefig("best_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_study(results_df, target_model="ConvE"):
    """
    绘制单一模型内部的超参数消融实验看板（科学严谨、莫兰迪配色）
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 过滤出目标模型的数据
    df_model = results_df[results_df['Model'] == target_model]
    if df_model.empty:
        print(f"[警告] 没有找到模型 {target_model} 的实验数据。")
        return

    # ---------------------------------------------------------
    # 图 A: Batch Size 与 Learning Rate 组合的 MRR 热力图
    # ---------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    # 取不同 dim 下的平均值来做热力图
    pivot_mrr = df_model.pivot_table(index='lr', columns='batch', values='Test MRR', aggfunc='mean')
    sns.heatmap(pivot_mrr, annot=True, fmt=".4f", cmap=sns.light_palette(MORANDI_PALETTE[0], as_cmap=True), ax=ax1, cbar_kws={'label': 'Mean Test MRR'})
    ax1.set_title(f"A. {target_model}: Batch Size vs Learning Rate (MRR)", fontsize=14, fontweight='bold', pad=10)
    ax1.invert_yaxis() # 让较小的 lr 在下方，符合直觉

    # ---------------------------------------------------------
    # 图 B: 维度 (Dim) 对模型指标的边际效应 (Pointplot)
    # ---------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    # 融合 MRR 和 Hits@10 以双轴显示
    sns.pointplot(data=df_model, x='dim', y='Test MRR', color=MORANDI_PALETTE[1], label='Test MRR', markers='o', ax=ax2)
    ax2.set_ylabel('Test MRR', color=MORANDI_PALETTE[1], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=MORANDI_PALETTE[1])
    
    ax2_twin = ax2.twinx()
    sns.pointplot(data=df_model, x='dim', y='Test Hits@10', color=MORANDI_PALETTE[2], label='Hits@10', markers='s', linestyles='--', ax=ax2_twin)
    ax2_twin.set_ylabel('Test Hits@10', color=MORANDI_PALETTE[2], fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=MORANDI_PALETTE[2])
    ax2.set_title(f"B. {target_model}: Impact of Embedding Dimension", fontsize=14, fontweight='bold', pad=10)

    # ---------------------------------------------------------
    # 图 C: 不同配置的性能分布箱线图 (评估超参敏感度)
    # ---------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(data=df_model, x='batch', y='Test MRR', hue='lr', palette=MORANDI_PALETTE[:len(df_model['lr'].unique())], ax=ax3)
    ax3.set_title(f"C. {target_model}: Performance Variance across Configurations", fontsize=14, fontweight='bold', pad=10)
    ax3.legend(title='Learning Rate', loc='lower right')

    # ---------------------------------------------------------
    # 图 D: 最佳 Top-5 配置雷达坐标系排布 (条形图展示)
    # ---------------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    top_5 = df_model.nlargest(5, 'Test MRR')
    # 构建配置的文本标签
    top_5['Config_Label'] = top_5.apply(lambda x: f"D:{int(x['dim'])} | B:{int(x['batch'])} | L:{x['lr']}", axis=1)
    
    sns.barplot(data=top_5, x='Test MRR', y='Config_Label', palette=sns.color_palette([MORANDI_PALETTE[4]]), ax=ax4)
    ax4.set_title(f"D. {target_model}: Top 5 Configurations Ranked by MRR", fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlim(df_model['Test MRR'].min() * 0.95, df_model['Test MRR'].max() * 1.05)
    ax4.set_ylabel("Configuration Details", fontsize=12)
    
    for p in ax4.patches:
        ax4.annotate(format(p.get_width(), '.4f'), 
                     (p.get_width(), p.get_y() + p.get_height() / 2.), 
                     ha='left', va='center', xytext=(5, 0), 
                     textcoords='offset points', fontsize=10)

    plt.suptitle(f"Hyperparameter Ablation Study: {target_model}", fontsize=18, y=0.98, fontweight='bold', color="#4A4A4A")
    plt.savefig(f"{target_model}_ablation_dashboard.png", dpi=300, bbox_inches='tight')
    print(f"\n[INFO] {target_model} 的消融实验看板已保存为 '{target_model}_ablation_dashboard.png'")
    plt.show()