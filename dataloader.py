import os
import numpy as np

class KGDataset:
    def __init__(self, data_path):
        # 读取实体和关系ID映射
        self.entity2id = self.read_mapping(os.path.join(data_path, 'entity2id.txt'))
        self.relation2id = self.read_mapping(os.path.join(data_path, 'relation2id.txt'))
        
        self.num_entity = len(self.entity2id)
        self.num_rel = len(self.relation2id)
        
        # 读取三元组
        self.train = self.read_triples(os.path.join(data_path, 'train.txt'))
        self.valid = self.read_triples(os.path.join(data_path, 'valid.txt'))
        self.test = self.read_triples(os.path.join(data_path, 'test.txt'))
        
        # 修复：numpy array 不能放入 set
        self.train_triples = set(tuple(tri) for tri in self.train)

    def read_mapping(self, path):
        mapping = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item, idx = line.strip().split()
                mapping[item] = int(idx)
        return mapping

    def read_triples(self, path):
        triples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.strip().split()
                h = self.entity2id[h]
                r = self.relation2id[r]
                t = self.entity2id[t]
                triples.append((h, r, t))
        return np.array(triples)
