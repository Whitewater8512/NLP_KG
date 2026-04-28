import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# 设置学术风格主题
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

DATASET_NAME = ""

# ==========================================
# 模块一：解析完整的训练日志并绘制学习曲线
# ==========================================
def plot_learning_curves(log_file_path):
    data = []
    current_config = None
    current_model = None
    
    # 1. 匹配类似 "[1/12] Running: TransE_D150_B32_L0.001" 提取模型名称
    config_pattern = re.compile(r'Running:\s+(\S+)')
    # 2. 匹配类似 "Epoch 005 | AvgLoss 5.1 | Valid MRR 0.2268" 提取训练动态
    epoch_pattern = re.compile(r'Epoch\s+(\d+)\s*\|\s*AvgLoss\s+([\d\.]+)\s*\|\s*Valid\s+MRR\s+([\d\.]+)', re.IGNORECASE)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 侦测当前所在的实验配置
                cfg_match = config_pattern.search(line)
                if cfg_match:
                    current_config = cfg_match.group(1)
                    current_model = current_config.split('_')[0] # 提取前缀作为大模型分类
                    continue
                
                # 如果在某个配置区块内，提取 epoch 训练指标
                if current_config:
                    ep_match = epoch_pattern.search(line)
                    if ep_match:
                        data.append({
                            'Model': current_model,
                            'Config': current_config,
                            'Epoch': int(ep_match.group(1)),
                            'Loss': float(ep_match.group(2)),
                            'Valid_MRR': float(ep_match.group(3))
                        })
                        
        df_log = pd.DataFrame(data)
        if df_log.empty:
            print("未能从日志中提取到数据，请检查正则表达式或文件内容。")
            return
            
        # 开始绘制收敛曲线
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 使用 seaborn 的 lineplot，它会自动将同一 Model 下的多组超参数表现取平均，并绘制阴影（置信区间）
        sns.lineplot(data=df_log, x='Epoch', y='Loss', hue='Model', ax=ax1, palette='Set2')
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Training Loss', fontweight='bold')
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1), title="Training Loss")

        # 实例化共享相同 x 轴的第二个坐标轴
        ax2 = ax1.twinx()  
        sns.lineplot(data=df_log, x='Epoch', y='Valid_MRR', hue='Model', ax=ax2, palette='Set2', linestyle='--')
        ax2.set_ylabel('Validation MRR', fontweight='bold')
        # 移走图例避免遮挡
        ax2.legend(loc='lower right', title="Valid MRR (Dashed)")

        plt.title('Training Dynamics by Model Family: Loss vs Validation MRR', fontsize=14, pad=15)
        fig.tight_layout()
        plt.savefig(f'{DATASET_NAME}_output/Learning_curves.png', dpi=300)
        plt.show()
        
    except FileNotFoundError:
        print(f"未找到日志文件: {log_file_path}")


# ==========================================
# 模块二：基于 CSV 的深层指标权衡与泛化分析
# ==========================================
def plot_csv_advanced_analysis(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图 1: 指标权衡 (Test MRR vs Test Hits@10)
    sns.scatterplot(
        data=df, x='Test MRR', y='Test Hits@10', 
        hue='Model', style='Model', s=150, palette='Set1', ax=axes[0], alpha=0.8
    )
    axes[0].set_title('Metric Trade-off: Precision vs Recall Capacity', fontsize=14)
    axes[0].set_xlabel('Test MRR (Focuses on Top-1)', fontweight='bold')
    axes[0].set_ylabel('Test Hits@10 (Focuses on Top-10)', fontweight='bold')
    
    # 子图 2: 泛化能力差异 (Best Valid MRR vs Test MRR)
    # 此处已修正为 Best Valid MRR 
    sns.scatterplot(
        data=df, x='Best Valid MRR', y='Test MRR', 
        hue='Model', s=100, palette='Set1', ax=axes[1], alpha=0.7
    )
    # 绘制 y=x 理想基准线
    min_val = min(df['Best Valid MRR'].min(), df['Test MRR'].min()) - 0.02
    max_val = max(df['Best Valid MRR'].max(), df['Test MRR'].max()) + 0.02
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y = x (Perfect Generalization)')
    
    axes[1].set_title('Generalization Gap: Validation vs Test Performance', fontsize=14)
    axes[1].set_xlabel('Best Validation MRR', fontweight='bold')
    axes[1].set_ylabel('Test MRR', fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{DATASET_NAME}_output/Generalization_gap.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    # DATASET_NAME = "WN18RR"
    DATASET_NAME = "FB15k-237"
    # 执行绘图，请确保这里的路径与你本地的路径保持一致
    # plot_learning_curves('WN18RR_output/train_log.txt')
    # plot_csv_advanced_analysis('WN18RR_output/all_experiments_results.csv')
    plot_learning_curves(f'{DATASET_NAME}_output/train_log.txt')
    plot_csv_advanced_analysis(f'{DATASET_NAME}_output/all_experiments_results.csv')
