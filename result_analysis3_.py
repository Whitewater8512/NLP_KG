import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from dataloader import KGDataset
from models import TransE, DistMult, ConvE, ComplEx
import warnings

# 屏蔽 NLTK 等不必要的警告（因为 FB15k 不需要 WordNet）
warnings.filterwarnings("ignore")

class BestModelVisualizerFB15k:
    def __init__(self, data_path, csv_path, weights_dir, device='cpu'):
        self.device = torch.device(device)
        self.weights_dir = weights_dir
        
        print(f"正在从 {data_path} 加载 FB15k-237 数据集...")
        self.data = KGDataset(data_path)
        self.id2ent = {v: k for k, v in self.data.entity2id.items()}
        self.id2rel = {v: k for k, v in self.data.relation2id.items()}
        
        # 1. 加载你下载好的 entity2text.txt
        self.ent2text = self._load_entity_mapping(data_path)
        
        # 2. 读取实验结果 CSV，锁定最佳模型
        print(f"正在分析实验日志 {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.best_models_info = self._get_best_models()
        
        self.output_dir = "KG_Visualizations_FB15k"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _load_entity_mapping(self, data_path):
        """解析本地 entity2text.txt 文件"""
        ent2text = {}
        mapping_file = os.path.join(data_path, 'entity2text.txt')
        
        if os.path.exists(mapping_file):
            print(f"成功检测到映射文件: {mapping_file}")
            count = 0
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # FB15k 映射表通常是 Tab 分隔：ID \t Name (\t Description)
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        ent2text[parts[0]] = parts[1]
                        count += 1
            print(f"已成功建立 {count} 个实体的文本映射映射库。")
        else:
            print(f"⚠️ 警告: 未在 {data_path} 下找到 entity2text.txt，节点将显示原始 ID。")
        return ent2text

    def _get_word(self, fb_id):
        """获取映射后的单词名称"""
        return self.ent2text.get(fb_id, fb_id)

    def _format_rel(self, rel_id):
        """简化关系路径，例如 /film/film/genre -> genre"""
        return rel_id.split('/')[-1] if '/' in rel_id else rel_id

    def _get_best_models(self):
        """从 CSV 挑选验证集 MRR 最高的配置"""
        best_info = []
        for model_name in ['TransE', 'DistMult', 'ConvE', 'ComplEx']:
            df_model = self.df[self.df['Model'] == model_name]
            if not df_model.empty:
                best_idx = df_model['Best Valid MRR'].idxmax()
                row = df_model.loc[best_idx]
                best_info.append({'name': model_name, 'id': row['Model_ID'], 'dim': int(row['dim'])})
                print(f"⭐ 最佳 {model_name} 配置已锁定: {row['Model_ID']}")
        return best_info

    def _load_weights(self, model_info):
        """加载对应模型的 PTH 权重文件"""
        model_classes = {'TransE': TransE, 'DistMult': DistMult, 'ConvE': ConvE, 'ComplEx': ComplEx}
        model = model_classes[model_info['name']](
            num_ent=self.data.num_entity, num_rel=self.data.num_rel, dim=model_info['dim']
        )
        weight_path = os.path.join(self.weights_dir, f"{model_info['id']}_best.pth")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=self.device))
            model.to(self.device).eval()
            return model
        return None

    def run_case_study(self, h_id, r_id, true_t_id, top_k=5):
        """执行案例分析并生成可视化成果物"""
        if h_id not in self.data.entity2id:
            print(f"错误: 实体 {h_id} 不在索引中。")
            return

        h_idx = self.data.entity2id[h_id]
        r_idx = self.data.relation2id[r_id]
        
        h_name = self._get_word(h_id)
        t_name = self._get_word(true_t_id)
        rel_short = self._format_rel(r_id)

        for info in self.best_models_info:
            print(f"\n正在通过模型 {info['name']} 推理预测...")
            model = self._load_weights(info)
            if not model: continue
            
            with torch.no_grad():
                h_t = torch.tensor([h_idx], device=self.device)
                r_t = torch.tensor([r_idx], device=self.device)
                logits = model(h_t, r_t).squeeze(0)
                scores, indices = torch.topk(logits, k=top_k)

            # 整理 Top-K 结果
            results = []
            for rank, (s, idx) in enumerate(zip(scores.cpu().numpy(), indices.cpu().numpy())):
                eid = self.id2ent[idx]
                results.append({'word': self._get_word(eid), 'score': s, 'is_correct': (eid == true_t_id), 'rank': rank+1})

            self._save_png(info['name'], h_name, rel_short, r_id, results)
            self._save_html(info['name'], h_name, rel_short, r_id, results)

    def _save_png(self, m_name, h_name, r_short, r_full, results):
        plt.figure(figsize=(10, 7))
        G = nx.DiGraph()
        G.add_node("C", label=h_name, color='#FF9999', size=3500)
        
        for res in results:
            nid = f"P_{res['rank']}"
            color = '#99FF99' if res['is_correct'] else '#E0E0E0'
            G.add_node(nid, label=res['word'], color=color, size=2000)
            G.add_edge("C", nid, label=f"{r_short}\nRank {res['rank']}")

        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        
        pos = nx.spring_layout(G, k=1.0, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=1.5)
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=9)
        
        plt.title(f"Model: {m_name} | Query: ({h_name}, {r_short}, ?)", fontsize=14)
        plt.axis('off')
        plt.savefig(os.path.join(self.output_dir, f"{m_name}_FB15k.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_html(self, m_name, h_name, r_short, r_full, results):
        net = Network(height='600px', width='100%', directed=True, notebook=False)
        net.force_atlas_2based()
        net.add_node("C", label=h_name, title="Head Entity", color='#FF6B6B', size=45)
        
        for res in results:
            nid = f"P_{res['rank']}"
            color = '#4ECDC4' if res['is_correct'] else '#C7F464'
            info = f"Rank: {res['rank']}\nScore: {res['score']:.4f}"
            if res['is_correct']: info += "\n(Ground Truth! ✅)"
            
            net.add_node(nid, label=res['word'], title=info, color=color, size=30)
            net.add_edge("C", nid, label=r_short, title=r_full, arrows='to')

        net.write_html(os.path.join(self.output_dir, f"{m_name}_FB15k.html"))

if __name__ == "__main__":
    # --- 请确保以下路径指向你 FB15k-237 的实际位置 ---
    vis = BestModelVisualizerFB15k(
        data_path='./FB15k-237', 
        csv_path='./FB15k-237_output/all_experiments_results.csv', 
        weights_dir='./FB15k-237_output/weights'
    )
    
    # 示例案例：星球大战 (Star Wars) -> 电影类型 (film/genre) -> 科幻 (Science Fiction)
    # 请根据你 entity2id.txt 里的真实 ID 进行替换
    vis.run_case_study(
        h_id="/m/0dtfn",           # Star Wars Episode IV
        r_id="/film/film/genre",    # Relation
        true_t_id="/m/06n90",       # Science Fiction
        top_k=10
    )
    print("\n✅ 完成！静态图与动态网页已存入 KG_Visualizations_FB15k 文件夹。")