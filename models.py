import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvE(nn.Module):
    def __init__(self, num_ent, num_rel, dim=200, label_smoothing=0.1):
        super().__init__()
        self.dim = dim
        
        self.h = 10
        self.w = self.dim // self.h
        assert self.dim % self.h == 0, f"Dimension {self.dim} must be divisible by {self.h}"
        
        self.label_smoothing = label_smoothing

        self.emb_e = nn.Embedding(num_ent, dim)
        self.emb_r = nn.Embedding(num_rel * 2, dim)

        nn.init.xavier_normal_(self.emb_e.weight)
        nn.init.xavier_normal_(self.emb_r.weight)

        self.inp_drop = nn.Dropout(0.2)
        self.hidden_drop = nn.Dropout(0.3)
        self.feat_drop = nn.Dropout(0.2)

        self.bn0 = nn.BatchNorm2d(1)
        self.conv = nn.Conv2d(1, 32, 3, 1, 0)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm1d(dim)
        
        out_h = (2 * self.h) - 3 + 1
        out_w = self.w - 3 + 1
        flat_size = 32 * out_h * out_w
        
        self.fc = nn.Linear(flat_size, dim)

    def forward(self, h, r):
        h_emb = self.emb_e(h).view(-1, 1, self.h, self.w)
        r_emb = self.emb_r(r).view(-1, 1, self.h, self.w)

        x = torch.cat([h_emb, r_emb], dim=2)
        x = self.bn0(x)
        x = self.inp_drop(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.hidden_drop(x)
        
        x = x.flatten(1)
        x = self.fc(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.feat_drop(x)
        
        return torch.matmul(x, self.emb_e.weight.t())

    def loss(self, logits, t):
        return F.cross_entropy(logits, t, label_smoothing=self.label_smoothing)

class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, dim=200, label_smoothing=0.1):
        super().__init__()
        self.emb_e = nn.Embedding(num_ent, dim)
        self.emb_r = nn.Embedding(num_rel * 2, dim) 
        nn.init.xavier_uniform_(self.emb_e.weight)
        nn.init.xavier_uniform_(self.emb_r.weight)
        self.label_smoothing = label_smoothing # 统一接收此参数

    def forward(self, h, r):
        eh = F.normalize(self.emb_e(h), p=2, dim=-1)
        er = self.emb_r(r)
        score = -torch.cdist(eh + er, F.normalize(self.emb_e.weight, p=2, dim=-1), p=1)
        return score
        
    def loss(self, logits, t):
        # 统一 Loss 接口
        return F.cross_entropy(logits, t, label_smoothing=self.label_smoothing)

class DistMult(nn.Module):
    def __init__(self, num_ent, num_rel, dim=200, label_smoothing=0.1):
        super().__init__()
        self.dim = dim
        self.label_smoothing = label_smoothing

        self.emb_e = nn.Embedding(num_ent, dim)
        self.emb_r = nn.Embedding(num_rel * 2, dim)

        nn.init.xavier_normal_(self.emb_e.weight)
        nn.init.xavier_normal_(self.emb_r.weight)

        self.inp_drop = nn.Dropout(0.2)
        self.rel_drop = nn.Dropout(0.2)

    def forward(self, h, r):
        h_emb = self.emb_e(h)
        r_emb = self.emb_r(r)

        h_emb = F.normalize(h_emb, p=2, dim=-1)
        h_emb = self.inp_drop(h_emb)
        r_emb = self.rel_drop(r_emb)

        query = h_emb * r_emb
        scores = torch.matmul(query, self.emb_e.weight.t())
        return scores

    def loss(self, logits, t):
        return F.cross_entropy(logits, t, label_smoothing=self.label_smoothing)

class ComplEx(nn.Module):
    def __init__(self, num_ent, num_rel, dim=200, label_smoothing=0.1):
        super().__init__()
        # 复数域需要实部和虚部，因此实际的 dim 是输入 dim 的一半，保证总参数量公平
        self.dim = dim // 2  
        self.label_smoothing = label_smoothing

        # 实体和关系的 实部 (Real) 与 虚部 (Imaginary)
        self.ent_re = nn.Embedding(num_ent, self.dim)
        self.ent_im = nn.Embedding(num_ent, self.dim)
        self.rel_re = nn.Embedding(num_rel * 2, self.dim)
        self.rel_im = nn.Embedding(num_rel * 2, self.dim)

        # 初始化 (极其关键)
        nn.init.xavier_normal_(self.ent_re.weight)
        nn.init.xavier_normal_(self.ent_im.weight)
        nn.init.xavier_normal_(self.rel_re.weight)
        nn.init.xavier_normal_(self.rel_im.weight)

        self.drop = nn.Dropout(0.2)

    def forward(self, h, r):
        # 1. 提取头实体和关系的复数向量
        h_re = self.drop(self.ent_re(h))
        h_im = self.drop(self.ent_im(h))
        r_re = self.drop(self.rel_re(r))
        r_im = self.drop(self.rel_im(r))

        # 2. 计算复数乘法 (h * r) 的实部和虚部
        # (h_re + i * h_im) * (r_re + i * r_im)
        q_re = h_re * r_re - h_im * r_im
        q_im = h_re * r_im + h_im * r_re

        # 3. 提取所有目标实体的共轭复数矩阵并转置 (用于 1-N 矩阵乘法)
        # 尾实体 t 的共轭复数为: t_re - i * t_im
        t_re = self.ent_re.weight.t()
        t_im = self.ent_im.weight.t()

        # 4. 计算最终得分 (取实部)
        # Score = Re( <h * r, \bar{t}> )
        score = torch.matmul(q_re, t_re) + torch.matmul(q_im, t_im)
        return score

    def loss(self, logits, t):
        return F.cross_entropy(logits, t, label_smoothing=self.label_smoothing)