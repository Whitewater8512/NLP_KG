import os
import torch
import numpy as np
from pyvis.network import Network
from dataloader import KGDataset
from models import TransE, DistMult, ConvE, ComplEx

class KGVisualizer:
    def __init__(self, dataset_path, model_class, config, device='cuda'):
        """
        config 需要包含: 'dim', 'batch', 'lr' 以定位权重文件
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data = KGDataset(dataset_path)
        
        # 1. 实例化模型架构
        self.model = model_class(
            num_ent=self.data.num_entity, 
            num_rel=self.data.num_rel, 
            dim=config['dim']
        ).to(self.device)
        
        # 2. 构造权重路径并加载
        model_name = model_class.__name__
        model_id = f"{model_name}_D{config['dim']}_B{config['batch']}_L{config['lr']}"
        weight_path = f"weights/{model_id}_best.pth"
        
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"未找到权重文件: {weight_path}，请确认训练已完成。")
            
        print(f"正在加载模型权重: {weight_path}")
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

        # 反向索引：ID -> 名称
        self.id2ent = {v: k for k, v in self.data.entity2id.items()}
        self.id2rel = {v: k for k, v in self.data.relation2id.items()}

    def predict_top_k(self, head_str, rel_str, k=10):
        """预测特定头实体和关系下的 Top-K 尾实体"""
        if head_str not in self.data.entity2id or rel_str not in self.data.relation2id:
            print(f"错误: 实体 '{head_str}' 或关系 '{rel_str}' 不存在。")
            return []

        h_idx = torch.tensor([self.data.entity2id[head_str]], device=self.device)
        r_idx = torch.tensor([self.data.relation2id[rel_str]], device=self.device)

        with torch.no_grad():
            logits = self.model(h_idx, r_idx)
            scores = torch.softmax(logits, dim=1) # 转化为概率分布
            top_probs, top_indices = torch.topk(scores, k=k)

        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                'tail': self.id2ent[idx.item()],
                'prob': prob.item()
            })
        return results

    def create_subgraph(self, center_entity, depth=1, top_k=5, filename="kg_result.html"):
        """
        以 center_entity 为中心，生成局部预测图谱
        depth: 扩散深度 (1表示只看邻居，2表示邻居的邻居)
        top_k: 每个节点预测时保留的最优边数
        """
        net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', directed=True)
        # 设置物理仿真参数，让节点自动排布更美观
        net.force_atlas_2based()

        added_nodes = set()
        queue = [(center_entity, 0)]
        
        while queue:
            curr_ent, curr_depth = queue.pop(0)
            if curr_depth >= depth: continue
            
            if curr_ent not in added_nodes:
                net.add_node(curr_ent, label=curr_ent, color='#9C7A6D', size=30 if curr_depth==0 else 20)
                added_nodes.add(curr_ent)

            # 遍历所有关系，寻找预测值最高的尾实体
            for rel_str in self.data.relation2id.keys():
                predictions = self.predict_top_k(curr_ent, rel_str, k=top_k)
                
                for pred in predictions:
                    tail = pred['tail']
                    prob = pred['prob']
                    
                    # 只展示概率较高的预测结果（阈值可调）
                    if prob > 0.01: 
                        if tail not in added_nodes:
                            net.add_node(tail, label=tail, color='#4A6478', size=15)
                            added_nodes.add(tail)
                            queue.append((tail, curr_depth + 1))
                        
                        net.add_edge(curr_ent, tail, title=f"Prob: {prob:.4f}", label=rel_str, value=prob)

        net.save_graph(filename)
        print(f"可视化网页已生成: {filename}")

if __name__ == '__main__':
    # 示例用法：
    # 假设你刚刚跑完的一个最佳配置是 ComplEx, dim=200, batch=128, lr=0.001
    my_config = {'dim': 200, 'batch': 128, 'lr': 0.001}
    
    viz = KGVisualizer(
        dataset_path='./WN18RR', 
        model_class=ComplEx, 
        config=my_config
    )

    # 1. 简单打印预测列表
    print("\n--- 模型预测结果 ---")
    results = viz.predict_top_k('dog_NN_1', '_hypernym', k=5)
    for r in results:
        print(f"Tail: {r['tail']:<20} | Confidence: {r['prob']:.4f}")

    # 2. 生成交互式网页图谱
    # 以 'dog_NN_1' 为中心，看看模型认为它有哪些上位词或相关概念
    viz.create_subgraph('dog_NN_1', depth=1, top_k=3, filename="dog_world.html")