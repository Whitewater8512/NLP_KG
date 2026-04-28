import os
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from dataloader import KGDataset
from models import TransE, DistMult, ConvE, ComplEx
import warnings

# 屏蔽 NLTK 警告
warnings.filterwarnings("ignore", category=UserWarning, module='nltk.corpus.reader.wordnet')
import nltk
from nltk.corpus import wordnet as wn

# ==========================================
# 辅助函数：ID 转 单词
# ==========================================
def get_word_from_id(synset_id):
    offset = int(synset_id)
    for pos in ['n', 'v', 'a', 'r']:
        try:
            synset = wn.synset_from_pos_and_offset(pos, offset)
            return synset.name().split('.')[0].replace('_', ' ')
        except:
            continue
    return str(synset_id)

# ==========================================
# 核心类：最佳模型提取与可视化分析器
# ==========================================
class BestModelVisualizer:
    def __init__(self, data_path, csv_path, weights_dir, device='cpu'):
        self.device = torch.device(device)
        self.weights_dir = weights_dir
        
        print(f"Loading dataset from {data_path}...")
        self.data = KGDataset(data_path)
        self.id2ent = {v: k for k, v in self.data.entity2id.items()}
        self.id2rel = {v: k for k, v in self.data.relation2id.items()}
        
        # 1. 读取 analysis2 中的逻辑，找出最佳模型
        print(f"Reading experiment results from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.best_models_info = self._get_best_models()
        
        # 创建输出文件夹
        self.output_dir = "KG_Visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_best_models(self):
        """按照 Best Valid MRR 从 CSV 中筛选出四个模型的最佳配置"""
        best_info = []
        for model_name in ['TransE', 'DistMult', 'ConvE', 'ComplEx']:
            df_model = self.df[self.df['Model'] == model_name]
            if not df_model.empty:
                # 寻找验证集表现最好的行
                best_idx = df_model['Best Valid MRR'].idxmax()
                best_row = df_model.loc[best_idx]
                best_info.append({
                    'name': model_name,
                    'id': best_row['Model_ID'],
                    'dim': int(best_row['dim'])
                })
                print(f"🏆 Found Best {model_name}: {best_row['Model_ID']} (Valid MRR: {best_row['Best Valid MRR']:.4f})")
        return best_info

    def _load_model(self, model_info):
        """根据信息加载模型结构和权重"""
        model_classes = {'TransE': TransE, 'DistMult': DistMult, 'ConvE': ConvE, 'ComplEx': ComplEx}
        model_class = model_classes[model_info['name']]
        
        model = model_class(num_ent=self.data.num_entity, num_rel=self.data.num_rel, dim=model_info['dim'])
        
        weight_path = os.path.join(self.weights_dir, f"{model_info['id']}_best.pth")
        if not os.path.exists(weight_path):
            print(f"⚠️ Warning: 找不到权重文件 {weight_path}")
            return None
            
        model.load_state_dict(torch.load(weight_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def visualize_prediction(self, center_ent, rel, true_tail, top_k=5):
        """为所有最佳模型生成特定三元组的预测可视化图谱"""
        
        if center_ent not in self.data.entity2id or rel not in self.data.relation2id:
            print("提供的实体或关系不在数据集中！")
            return
            
        h_idx = self.data.entity2id[center_ent]
        r_idx = self.data.relation2id[rel]
        
        center_word = get_word_from_id(center_ent)
        true_tail_word = get_word_from_id(true_tail)
        rel_word = rel.replace('_', '')
        
        for info in self.best_models_info:
            print(f"\nProcessing {info['name']}...")
            model = self._load_model(info)
            if model is None: continue
            
            # 进行推理预测
            h_tensor = torch.tensor([h_idx], dtype=torch.long, device=self.device)
            r_tensor = torch.tensor([r_idx], dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                logits = model(h_tensor, r_tensor).squeeze(0)
                scores, indices = torch.topk(logits, k=top_k)
            
            # 准备绘图数据
            predictions = []
            for rank, (score, idx) in enumerate(zip(scores.cpu().numpy(), indices.cpu().numpy())):
                pred_ent_id = self.id2ent[idx]
                pred_word = get_word_from_id(pred_ent_id)
                is_true = (pred_ent_id == true_tail)
                predictions.append((pred_word, score, is_true, rank + 1))
                
            self._draw_static_png(info['name'], center_word, rel_word, true_tail_word, predictions)
            self._draw_dynamic_html(info['name'], center_word, rel_word, true_tail_word, predictions)

    def _draw_static_png(self, model_name, center_word, rel_word, true_tail_word, predictions):
        """生成并保存静态 PNG 图片"""
        plt.figure(figsize=(10, 8))
        G = nx.DiGraph()
        
        # 使用 CENTER 作为底层 ID，避免同名冲突
        G.add_node("CENTER", label=center_word, color='#FF9999', size=3000)
        
        for pred_word, score, is_true, rank in predictions:
            # 给每个预测节点一个唯一的 ID (例如 PRED_1, PRED_2)
            node_id = f"PRED_{rank}"
            color = '#99FF99' if is_true else '#E0E0E0'
            G.add_node(node_id, label=pred_word, color=color, size=2000)
            G.add_edge("CENTER", node_id, label=f"{rel_word}\n(Rank {rank})")

        # 动态提取属性，保证与 G.nodes() 顺序和数量绝对一致
        node_colors = [G.nodes[n]['color'] for n in G.nodes()]
        node_sizes = [G.nodes[n]['size'] for n in G.nodes()]
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}

        pos = nx.spring_layout(G, k=0.9, seed=42)
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, edgecolors='black')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='gray', width=2)
        nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=10)
        
        plt.title(f"{model_name} Knowledge Graph Prediction\nTarget: (?, {rel_word}, {center_word})", fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        
        out_path = os.path.join(self.output_dir, f"{model_name}_prediction.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  -> Saved PNG: {out_path}")

    def _draw_dynamic_html(self, model_name, center_word, rel_word, true_tail_word, predictions):
        """生成并保存动态 HTML (XML格式封装) 页面"""
        net = Network(height='700px', width='100%', directed=True, notebook=False)
        net.force_atlas_2based()
        
        # 同样使用唯一 ID 避免 PyVis 内部的渲染冲突
        net.add_node("CENTER", label=center_word, title="Center Entity", color='#FF6B6B', size=40, font={'size': 20, 'face': 'Arial', 'bold': True})
        
        for pred_word, score, is_true, rank in predictions:
            node_id = f"PRED_{rank}"
            color = '#4ECDC4' if is_true else '#C7F464'
            title = f"Score: {score:.4f} | Rank: {rank}"
            if is_true:
                title += " | [Ground Truth]"
                
            net.add_node(node_id, label=pred_word, title=title, color=color, size=25)
            net.add_edge("CENTER", node_id, title=rel_word, label=f"{rel_word} (R:{rank})", arrows='to', color='#555555')

        net.show_buttons(filter_=['physics'])
        out_path = os.path.join(self.output_dir, f"{model_name}_prediction.html")
        net.write_html(out_path)
        print(f"  -> Saved HTML: {out_path}")

if __name__ == "__main__":
    # ==========================================
    # 路径配置区
    # ==========================================
    DATASET_PATH = './WN18RR'                             # WN18RR 数据集路径
    CSV_PATH = './WN18RR_output/all_experiments_results.csv'            # 你跑出来的 CSV 结果文件
    WEIGHTS_DIR = './WN18RR_output/weights'                             # 存放所有 .pth 文件的文件夹

    # 初始化分析器并执行可视化
    visualizer = BestModelVisualizer(
        data_path=DATASET_PATH, 
        csv_path=CSV_PATH, 
        weights_dir=WEIGHTS_DIR,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    
    # 设定你要分析的案例 (头实体ID, 关系ID, 真实尾实体ID)
    # 此处的例子： '00260881' (cat), '_hypernym' (上位词), '00260622' (feline)
    TEST_HEAD = "00260881"
    TEST_REL = "_hypernym"
    TEST_TAIL = "00260622"
    
    print("\n" + "="*50)
    print("开始为四个最佳模型生成知识图谱预测可视化...")
    print("="*50)
    
    # top_k=5 表示在图谱中画出模型认为最有可能的 5 个连接点
    visualizer.visualize_prediction(
        center_ent=TEST_HEAD, 
        rel=TEST_REL, 
        true_tail=TEST_TAIL, 
        top_k=10
    )
    
    print("\n✅ 所有模型的静态 PNG 和动态 HTML 均已生成完毕，存放在 'KG_Visualizations' 文件夹中！")