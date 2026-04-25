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

    experiment_grid = {
        'TransE': {'dim': [100, 150, 200, 250], 'batch': [32, 64, 128], 'epoch': [10], 'lr': [0.001, 0.0005, 0.0001], 'eval_batch_size': [128], 'eval_freq': [10]},
        'DistMult': {'dim': [100, 150, 200, 250], 'batch': [32, 64, 128], 'epoch': [10], 'lr': [0.001, 0.0005, 0.0001], 'eval_batch_size': [128], 'eval_freq': [10]},
        'ConvE': {'dim': [100, 150, 200, 250], 'batch': [32, 64, 128], 'epoch': [10], 'lr': [0.001, 0.0005, 0.0001], 'eval_batch_size': [128], 'eval_freq': [10]}
    }

    final_results = []
    histories_dict = {}

    for model_name, grid in experiment_grid.items():
        keys, values = zip(*grid.items())
        permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        # 为了演示，我们取每个模型的第一个配置进行跑和画图
        config = permutations[0] 
        print(f"\n{'='*50}")
        print(f"Running Experiment: {model_name}")
        print(f"Config: {config}")
        print(f"{'='*50}")
        
        model = model_classes[model_name](
            num_ent=data.num_entity, 
            num_rel=data.num_rel, 
            dim=config['dim']
        )
        
        trainer = KGTrainer(model, data, device, config)
        
        # 接收三个返回值：测试MRR，测试H10，训练历史曲线
        test_mrr, test_h10, history = trainer.train(verbose=True) 
        
        # 记录用于表格和绘图的数据
        record = {
            'Model': model_name,
            **config,
            'Test MRR': round(test_mrr, 4),
            'Test Hits@10': round(test_h10, 4)
        }
        final_results.append(record)
        histories_dict[model_name] = history # 保存历史曲线

    # 1. 打印实验结果汇总表格
    df = pd.DataFrame(final_results)
    print("\n\n" + "#"*40)
    print("### 实验结果汇总 ###")
    print("#"*40)
    print(df.to_markdown(index=False))

    # 可视化实验结果
    viz.plotize_experiment_results()

if __name__ == '__main__':
    run_experiments()