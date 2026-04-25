import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['TQDM_DISABLE'] = '1'  # 禁用 tqdm 输出，保持日志清晰

import random
import numpy as np
import torch
import itertools
import pandas as pd
import visualization as viz
from dataloader import KGDataset
from models import TransE, DistMult, ConvE, ComplEx
from trainer import KGTrainer

def set_seed(seed=42):
    """固定所有随机种子，确保实验完全可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_experiments(dataset_path='./WN18RR'):
    set_seed(42)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Loading data from {dataset_path} ...")
    data = KGDataset(dataset_path)
    
    model_classes = {
        'TransE': TransE,
        'DistMult': DistMult,
        'ConvE': ConvE,
        'ComplEx': ComplEx
    }

    experiment_grid = {
        'TransE': {
            'dim': [150, 200, 250], 
            'batch': [32, 64],  
            'epoch': [100], 
            'lr': [0.001, 0.0005],
            'eval_batch_size': [16], 
            'eval_freq': [5],
            'patience': [5]
        },
        'DistMult': {
            'dim': [150, 200, 250], 
            'batch': [128, 256], 
            'epoch': [100], 
            'lr': [0.005, 0.001], 
            'eval_batch_size': [64], 
            'eval_freq': [5],
            'patience': [5]
        },
        'ConvE': {
            'dim': [150, 200, 250], 
            'batch': [128, 256], 
            'epoch': [100], 
            'lr': [0.005, 0.001], 
            'eval_batch_size': [64], 
            'eval_freq': [5],
            'patience': [5]
        },
        'ComplEx': {
            'dim': [150, 200, 250], 
            'batch': [128, 256], 
            'epoch': [100], 
            'lr': [0.005, 0.001],
            'eval_batch_size': [64], 
            'eval_freq': [5],
            'patience': [5]
        }
    }

    all_results = []
    all_histories = {}

    # 第一阶段：执行完整的超参数网格搜索
    for model_name, grid in experiment_grid.items():
        keys, values = zip(*grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"\n🚀 开始 {model_name} 的超参数网格搜索，共计 {len(permutations)} 组配置...")
        
        for idx, config in enumerate(permutations):
            model_id = f"{model_name}_D{config['dim']}_B{config['batch']}_L{config['lr']}"
            
            print(f"\n{'='*60}")
            print(f"[{idx+1}/{len(permutations)}] Running: {model_id}")
            print(f"{'='*60}")
            
            # 每次初始化模型前再次固定种子，防止由于不同配置跳过 early stopping 导致后续随机序列错位
            set_seed(42 + idx) 
            
            model = model_classes[model_name](
                num_ent=data.num_entity, 
                num_rel=data.num_rel, 
                dim=config['dim']
            )
            
            trainer = KGTrainer(model, data, device, config)
            test_mrr, test_h10, history = trainer.train(verbose=True) 
            
            # 提取历史记录中最佳的 Valid MRR
            valid_mrrs = [m for m in history['valid_mrr'] if m is not None]
            best_valid_mrr = max(valid_mrrs) if valid_mrrs else 0.0

            record = {
                'Model': model_name,
                'Model_ID': model_id,
                **config,
                'Best Valid MRR': round(best_valid_mrr, 4), # 记录验证集表现
                'Test MRR': round(test_mrr, 4),
                'Test Hits@10': round(test_h10, 4)
            }
            all_results.append(record)
            all_histories[model_id] = history

    df_all = pd.DataFrame(all_results)
    df_all.to_csv("all_experiments_results.csv", index=False)

    # 第二阶段：严谨筛选——基于【验证集最佳表现】选择最终架构
    best_results = []
    best_histories_dict = {}
    
    for model_name in model_classes.keys():
        df_model = df_all[df_all['Model'] == model_name]
        
        if not df_model.empty:
            # 修改点：根据 Best Valid MRR 找最大值，而不是 Test MRR
            best_idx = df_model['Best Valid MRR'].idxmax()
            best_row = df_model.loc[best_idx]
            
            best_model_id = best_row['Model_ID']
            best_results.append(best_row.to_dict())
            best_histories_dict[best_model_id] = all_histories[best_model_id]

    best_df = pd.DataFrame(best_results)
    print("\n🏆 各模型基于验证集筛选出的最佳配置及其最终测试表现：")
    print(best_df[['Model_ID', 'Best Valid MRR', 'Test MRR', 'Test Hits@10', 'dim', 'batch', 'lr']].to_markdown(index=False))
    
    # 传递给可视化模块
    viz.plot_model_comparison(best_results, best_histories_dict)

    print("\n📊 生成各模型消融实验分析图...")
    for model_name in model_classes.keys():
        viz.plot_ablation_study(df_all, target_model=model_name)

if __name__ == '__main__':
    run_experiments()