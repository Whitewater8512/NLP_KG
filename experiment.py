import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import itertools
import pandas as pd
from dataloader import KGDataset
from models import TransE, DistMult, ConvE
from trainer import KGTrainer

def run_experiments(dataset_path='./WN18RR'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading data from {dataset_path} ...")
    data = KGDataset(dataset_path)
    
    # 定义模型字典
    model_classes = {
        'TransE': TransE,
        'DistMult': DistMult,
        'ConvE': ConvE
    }

    experiment_grid = {
        'TransE': {'dim': [200], 'batch': [64], 'epoch': [100], 'lr': [0.001], 'eval_batch_size': [4]},
        'DistMult': {'dim': [200], 'batch': [128], 'epoch': [100], 'lr': [0.001], 'eval_batch_size': [128]},
        'ConvE': {'dim': [200], 'batch': [128], 'epoch': [150], 'lr': [0.001], 'eval_batch_size': [128]}
    }

    results = []

    for model_name, grid in experiment_grid.items():
        # 生成当前模型所有可能的超参组合
        keys, values = zip(*grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for config in permutations:
            print(f"\n{'='*50}")
            print(f"Running Experiment: {model_name}")
            print(f"Config: {config}")
            print(f"{'='*50}")
            
            # 初始化模型
            model = model_classes[model_name](
                num_ent=data.num_entity, 
                num_rel=data.num_rel, 
                dim=config['dim']
            )
            
            # 初始化 Trainer
            trainer = KGTrainer(model, data, device, config)
            
            # 运行训练并获取测试结果
            # verbose=False 可以关闭每个 epoch 的打印，保持终端干净
            test_mrr, test_h10 = trainer.train(verbose=True) 
            
            # 记录结果
            record = {
                'Model': model_name,
                **config,
                'Test MRR': round(test_mrr, 4),
                'Test Hits@10': round(test_h10, 4)
            }
            results.append(record)

    # 导出实验报告表格
    df = pd.DataFrame(results)
    print("\n\n" + "#"*40)
    print("### 实验结果汇总 ###")
    print("#"*40)
    print(df.to_markdown(index=False))
    
    # 也可以保存为 csv 供绘图使用
    # df.to_csv('experiment_results.csv', index=False)

if __name__ == '__main__':
    run_experiments()