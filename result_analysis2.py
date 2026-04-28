import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATASET_NAME = ""

def plot_result_analysis():
    df = pd.read_csv(f'{DATASET_NAME}_output/all_experiments_results.csv')
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

    models = ['TransE', 'DistMult', 'ConvE', 'ComplEx']
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(18, 22))

    for i, model in enumerate(models):
        model_df = df[df['Model'] == model]
        
        # Col 1: Dim Impact
        ax1 = axes[i, 0]
        sns.pointplot(x='dim', y='Test MRR', data=model_df, ax=ax1, markers='o', color='darkslateblue')
        summary_dim = model_df.groupby('dim')['Test MRR'].mean().reset_index()
        dims = sorted(model_df['dim'].unique())
        for _, row in summary_dim.iterrows():
            x_pos = dims.index(row['dim'])
            ax1.text(x_pos, row['Test MRR'], f'{row["Test MRR"]:.4f}', ha='center', va='bottom', fontsize=8, color='black')
        ax1.set_title(f'{model}: MRR vs. Dimension', fontsize=12)
        ax1.set_xlabel('')
        ax1.set_ylabel('Test MRR')

        # Col 2: Batch Distribution
        ax2 = axes[i, 1]
        sns.boxplot(x='batch', y='Test MRR', data=model_df, ax=ax2, palette='Set2')
        medians_batch = model_df.groupby('batch')['Test MRR'].median().values
        for j, med in enumerate(medians_batch):
            ax2.text(j, med, f'{med:.4f}', ha='center', va='bottom', fontsize=8, color='black', fontweight='bold')
        ax2.set_title(f'{model}: MRR by Batch Size', fontsize=12)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('')

        # Col 3: LR Heatmap (if enough data)
        ax3 = axes[i, 2]
        try:
            pivot_df = model_df.pivot_table(index='batch', columns='lr', values='Test MRR', aggfunc='mean')
            sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.4f', linewidths=.5, ax=ax3, cbar_kws={'shrink': 0.8})
            ax3.set_title(f'{model}: MRR Heatmap (Batch vs LR)', fontsize=12)
            ax3.set_xlabel('Learning Rate')
            ax3.set_ylabel('Batch Size')
        except:
            ax3.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=12)
            ax3.set_title(f'{model}: Heatmap (N/A)', fontsize=12)
            ax3.axis('off')

    plt.tight_layout()
    plt.savefig(f'{DATASET_NAME}_output/result_analysis2.png', dpi=300)
    plt.show()

DATASET_NAME = "FB15k-237"
plot_result_analysis()