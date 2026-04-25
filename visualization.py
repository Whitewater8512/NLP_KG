import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

MORANDI_PALETTE = ["#4A6478", "#9C7A6D", "#586F6B", "#8F7179", "#B4956D", "#635C51", "#3E5254"]

def plot_model_comparison(best_results_df, best_histories_dict):
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.25)
    
    models = list(best_histories_dict.keys())
    
    # ---------------------------------------------------------
    # 图 A: 最佳配置的训练 Loss 收敛情况
    # ---------------------------------------------------------
    ax1 = plt.subplot(2, 2, 1)
    for i, model_id in enumerate(models):
        hist = best_histories_dict[model_id]
        ax1.plot(hist['epoch'], hist['train_loss'], label=model_id, 
                 color=MORANDI_PALETTE[i], lw=2.5, alpha=0.8)
        
        if hist['epoch']:
            final_epoch = hist['epoch'][-1]
            final_loss = hist['train_loss'][-1]
            ax1.text(final_epoch, final_loss, f" {final_loss:.3f}", 
                     color=MORANDI_PALETTE[i], fontsize=10, va='center', fontweight='bold')

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
        
        if valid_epochs:
            ax2.plot(valid_epochs, valid_mrrs, label=model_id, 
                     color=MORANDI_PALETTE[i], marker='o', markersize=4, lw=2)
            
            max_mrr = max(valid_mrrs)
            max_idx = valid_mrrs.index(max_mrr)
            max_epoch = valid_epochs[max_idx]
            ax2.annotate(f"{max_mrr:.4f}", 
                         xy=(max_epoch, max_mrr), xytext=(0, 5), textcoords="offset points",
                         ha='center', va='bottom', color=MORANDI_PALETTE[i], fontsize=10, fontweight='bold')

    ax2.set_title("B. Best Config: Validation MRR Progress", fontsize=14, pad=10, fontweight='bold')
    ax2.set_ylabel("Validation MRR", fontsize=12)
    ax2.legend(fontsize=9)

    # ---------------------------------------------------------
    # 图 C: 最佳配置的计算效率对比
    # ---------------------------------------------------------
    ax3 = plt.subplot(2, 2, 3)
    efficiency_data = []
    for model_id in models:
        hist = best_histories_dict[model_id]
        avg_train_time = np.mean(hist['train_time'])
        final_mrr = next(item['Test MRR'] for item in best_results_df if item['Model_ID'] == model_id)
        efficiency_data.append({'Model': model_id, 'Train Time/Epoch (s)': avg_train_time, 'MRR': final_mrr})
    
    df_eff = pd.DataFrame(efficiency_data)
    sns.scatterplot(data=df_eff, x='Train Time/Epoch (s)', y='MRR', hue='Model', 
                    palette=MORANDI_PALETTE[:len(models)], s=200, ax=ax3, legend=False)
    
    for _, row in df_eff.iterrows():
        ax3.text(row['Train Time/Epoch (s)'], row['MRR'], 
                 f"  {row['Model'].split('_')[0]}\n  ({row['Train Time/Epoch (s)']:.1f}s, {row['MRR']:.4f})", 
                 color='#333333', fontsize=10, va='center', ha='left')

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
    
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.4f', padding=3, fontsize=10, fontweight='bold', color="#4A4A4A")

    ax4.set_title("D. Final Best-in-Class Performance", fontsize=14, pad=10, fontweight='bold')
    
    plt.suptitle("KGE Excellence: Comparison of Best Performing Configurations", 
                 fontsize=18, y=0.98, fontweight='bold', color="#4A4A4A")
    plt.savefig("best_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_ablation_study(results_df, target_model="ConvE"):
    """
    绘制单一模型内部的超参数消融实验看板
    """
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig = plt.figure(figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    df_model = results_df[results_df['Model'] == target_model]
    if df_model.empty:
        print(f"[警告] 没有找到模型 {target_model} 的实验数据。")
        return

    # 图 A: 热力图
    ax1 = plt.subplot(2, 2, 1)
    pivot_mrr = df_model.pivot_table(index='lr', columns='batch', values='Test MRR', aggfunc='mean')
    sns.heatmap(pivot_mrr, annot=True, fmt=".4f", cmap=sns.light_palette(MORANDI_PALETTE[0], as_cmap=True), ax=ax1, cbar_kws={'label': 'Mean Test MRR'})
    ax1.set_title(f"A. {target_model}: Batch Size vs Learning Rate (MRR)", fontsize=14, fontweight='bold', pad=10)
    ax1.invert_yaxis()

    # ---------------------------------------------------------
    # 图 B: 维度 (Dim) 边际效应 
    # ---------------------------------------------------------
    ax2 = plt.subplot(2, 2, 2)
    sns.pointplot(data=df_model, x='dim', y='Test MRR', color=MORANDI_PALETTE[1], label='Test MRR', markers='o', ax=ax2)
    ax2.set_ylabel('Test MRR', color=MORANDI_PALETTE[1], fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=MORANDI_PALETTE[1])
    
    ax2_twin = ax2.twinx()
    sns.pointplot(data=df_model, x='dim', y='Test Hits@10', color=MORANDI_PALETTE[2], label='Hits@10', markers='s', linestyles='--', ax=ax2_twin)
    ax2_twin.set_ylabel('Test Hits@10', color=MORANDI_PALETTE[2], fontweight='bold')
    ax2_twin.tick_params(axis='y', labelcolor=MORANDI_PALETTE[2])
    ax2.set_title(f"B. {target_model}: Impact of Embedding Dimension", fontsize=14, fontweight='bold', pad=10)

    dim_stats = df_model.groupby('dim')[['Test MRR', 'Test Hits@10']].mean().reset_index()
    for idx, row in dim_stats.iterrows():        # 标注 MRR (在点下方)
        ax2.text(idx, row['Test MRR'], f" {row['Test MRR']:.4f}", 
                 color=MORANDI_PALETTE[1], ha='center', va='top', fontsize=10, fontweight='bold')
        ax2_twin.text(idx, row['Test Hits@10'], f" {row['Test Hits@10']:.4f}\n", 
                      color=MORANDI_PALETTE[2], ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 图 C: 箱线图 (用来观察分布方差，通常不在箱线上堆叠具体数值以免杂乱)
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(data=df_model, x='batch', y='Test MRR', hue='lr', palette=MORANDI_PALETTE[:len(df_model['lr'].unique())], ax=ax3)
    ax3.set_title(f"C. {target_model}: Performance Variance across Configurations", fontsize=14, fontweight='bold', pad=10)
    ax3.legend(title='Learning Rate', loc='lower right')

    # 图 D: Top 5 条形图
    ax4 = plt.subplot(2, 2, 4)
    top_5 = df_model.nlargest(5, 'Test MRR')
    top_5['Config_Label'] = top_5.apply(lambda x: f"D:{int(x['dim'])} | B:{int(x['batch'])} | L:{x['lr']}", axis=1)