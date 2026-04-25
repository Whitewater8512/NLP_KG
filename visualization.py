import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

MORANDI_PALETTE = ["#949483", "#B1A494", "#A1B3B3", "#D6C7C7", "#9B8E7D", "#E5E1D8", "#7A8B8B"]

def plot_comprehensive_dashboard(final_results, histories_dict):
    """
    绘制专业的 4 宫格实验对比看板
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    models = list(histories_dict.keys())
    
    # ---------------------------------------------------------
    # 图 A: 训练 Loss 曲线 (对数坐标系以平滑差异)
    # ---------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    for i, model in enumerate(models):
        hist = histories_dict[model]
        ax1.plot(hist['epoch'], hist['train_loss'], label=model, 
                 color=MORANDI_PALETTE[i], lw=2.5, alpha=0.9)
    ax1.set_title("A. Training Loss Convergence", fontsize=14, pad=10, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Total Loss", fontsize=12)
    ax1.legend()

    # ---------------------------------------------------------
    # 图 B: 验证集 MRR 收敛曲线 (处理 None 值)
    # ---------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    for i, model in enumerate(models):
        hist = histories_dict[model]
        # 过滤掉 None 的点（只在 eval_freq 的 epoch 有数据）
        valid_epochs = [e for e, mrr in zip(hist['epoch'], hist['valid_mrr']) if mrr is not None]
        valid_mrrs = [mrr for mrr in hist['valid_mrr'] if mrr is not None]
        
        ax2.plot(valid_epochs, valid_mrrs, label=model, 
                 color=MORANDI_PALETTE[i], marker='o', markersize=5, lw=2)
    ax2.set_title("B. Validation MRR over Epochs", fontsize=14, pad=10, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Validation MRR", fontsize=12)
    ax2.legend()

    # ---------------------------------------------------------
    # 图 C: 时间效率对比 (平均每个 Epoch 的训练/验证时间)
    # ---------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    avg_times = []
    for model in models:
        hist = histories_dict[model]
        avg_train = np.mean(hist['train_time'])
        # 排除 0.0 的评估时间
        real_eval_times = [t for t in hist['eval_time'] if t > 0]
        avg_eval = np.mean(real_eval_times) if real_eval_times else 0
        avg_times.append({'Model': model, 'Type': 'Train Time/Epoch', 'Seconds': avg_train})
        avg_times.append({'Model': model, 'Type': 'Eval Time/Eval', 'Seconds': avg_eval})
        
    df_time = pd.DataFrame(avg_times)
    sns.barplot(data=df_time, x='Model', y='Seconds', hue='Type', 
                palette=[MORANDI_PALETTE[2], MORANDI_PALETTE[4]], ax=ax3)
    ax3.set_title("C. Computational Cost Analysis", fontsize=14, pad=10, fontweight='bold')
    ax3.set_ylabel("Seconds", fontsize=12)
    
    # ---------------------------------------------------------
    # 图 D: 最终测试集综合表现 (Bar Plot)
    # ---------------------------------------------------------
    ax4 = plt.subplot(2, 2, 4)
    df_metrics = pd.DataFrame(final_results)
    df_melted = df_metrics.melt(id_vars='Model', value_vars=['Test MRR', 'Test Hits@10'], 
                                var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='Metric', y='Score', hue='Model', 
                palette=MORANDI_PALETTE[:len(models)], ax=ax4)
    ax4.set_title("D. Final Test Set Performance", fontsize=14, pad=10, fontweight='bold')
    ax4.set_ylim(0, max(df_melted['Score']) * 1.2) # 留出顶部空间显示图例
    
    # 添加数值标签
    for p in ax4.patches:
        ax4.annotate(format(p.get_height(), '.3f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', xytext = (0, 8), 
                     textcoords = 'offset points', fontsize=10)

    # 总体标题
    plt.suptitle("Knowledge Graph Embedding: Model Comparison & Efficiency Analysis", 
                 fontsize=18, y=0.98, fontweight='bold', color="#4A4A4A")
    
    # 保存高清图表
    plt.savefig("experiment_dashboard.png", dpi=300, bbox_inches='tight')
    print("\n[INFO] 可视化看板已保存为 'experiment_dashboard.png'")
    plt.show()