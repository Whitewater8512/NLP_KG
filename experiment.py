import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import itertools
import pandas as pd
import visualization as viz
from dataloader import KGDataset
from models import TransE, DistMult, ConvE
from trainer import KGTrainer

def run_experiments(dataset_path='./WN18RR'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading data from {dataset_path} ...")
    data = KGDataset(dataset_path)
    
    model_classes = {
        'TransE': TransE,
        'DistMult': DistMult,
        'ConvE': ConvE
    }

    # experiment_grid = {
    #     'TransE': {'dim': [100, 150, 200, 250], 'batch': [32, 64, 128], 'epoch': [100], 'lr': [0.001, 0.0005, 0.0001], 'eval_batch_size': [128], 'eval_freq': [5]},
    #     'DistMult': {'dim': [100, 150, 200, 250], 'batch': [32, 64, 128], 'epoch': [100], 'lr': [0.001, 0.0005, 0.0001], 'eval_batch_size': [128], 'eval_freq': [5]},
    #     'ConvE': {'dim': [100, 150, 200, 250], 'batch': [32, 64, 128], 'epoch': [100], 'lr': [0.001, 0.0005, 0.0001], 'eval_batch_size': [128], 'eval_freq': [5]}
    # }

    # DEBUG: 用于快速实验展示最终效果
    experiment_grid = {
        'TransE': {'dim': [150, 200], 'batch': [64], 'epoch': [5], 'lr': [0.001], 'eval_batch_size': [128], 'eval_freq': [1]},
        'DistMult': {'dim': [150, 200], 'batch': [64], 'epoch': [5], 'lr': [0.001], 'eval_batch_size': [128], 'eval_freq': [1]},
        'ConvE': {'dim': [150, 200], 'batch': [64], 'epoch': [5], 'lr': [0.001], 'eval_batch_size': [128], 'eval_freq': [1]}
    }

    all_results = []
    all_histories = {} # 使用 Model_ID 作为 key 存储所有曲线

    # 第一阶段：执行完整的超参数网格搜索
    for model_name, grid in experiment_grid.items():
        keys, values = zip(*grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        print(f"\n🚀 开始 {model_name} 的超参数网格搜索，共计 {len(permutations)} 组配置...")
        
        for idx, config in enumerate(permutations):
            # 构建唯一的配置标识符 (Model_ID)
            model_id = f"{model_name}_D{config['dim']}_B{config['batch']}_L{config['lr']}"
            
            print(f"\n{'='*50}")
            print(f"[{idx+1}/{len(permutations)}] Running: {model_id}")
            print(f"{'='*50}")
            
            model = model_classes[model_name](
                num_ent=data.num_entity, 
                num_rel=data.num_rel, 
                dim=config['dim']
            )
            
            trainer = KGTrainer(model, data, device, config)
            
            test_mrr, test_h10, history = trainer.train(verbose=True) 
            
            record = {
                'Model': model_name,
                'Model_ID': model_id,
                **config,
                'Test MRR': round(test_mrr, 4),
                'Test Hits@10': round(test_h10, 4)
            }
            all_results.append(record)
            all_histories[model_id] = history # 绑定对应曲线

    # 打印全局结果并保存
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("all_experiments_results.csv", index=False)

    # 第二阶段：自动筛选每个模型的【最佳配置】
    best_results = []
    best_histories_dict = {}
    
    for model_name in model_classes.keys():
        # 获取该模型的所有实验结果
        df_model = df_all[df_all['Model'] == model_name]
        
        if not df_model.empty:
            # 找到 Test MRR 最大的那一行
            best_idx = df_model['Test MRR'].idxmax()
            best_row = df_model.loc[best_idx]
            
            # 将最佳结果和对应的曲线提取出来
            best_model_id = best_row['Model_ID']
            best_results.append(best_row.to_dict())
            best_histories_dict[best_model_id] = all_histories[best_model_id]

    # 打印最终的“华山论剑”对阵表
    best_df = pd.DataFrame(best_results)
    print(best_df[['Model_ID', 'Test MRR', 'Test Hits@10', 'dim', 'batch', 'lr']].to_markdown(index=False))
    
    # 将精选出的最佳结果交给新的可视化函数
    viz.plot_model_comparison(best_results, best_histories_dict)

    print("\n生成各模型消融实验分析图...")
    for model_name in model_classes.keys():
        viz.plot_ablation_study(df_all, target_model=model_name)

if __name__ == '__main__':
    run_experiments()