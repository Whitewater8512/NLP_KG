import time
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class KGTrainer:
    def __init__(self, model, data, device, config):
        self.model = model.to(device)
        if int(torch.__version__.split('.')[0]) >= 2:
            self.model = torch.compile(self.model)
        self.data = data
        self.device = device
        self.config = config
        
        self.num_rel = data.num_rel
        self.all_true = self._get_all_true_triples()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'], fused=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epoch'], eta_min=1e-6
        )
        self.train_data = torch.tensor(self.data.train, dtype=torch.long, device=self.device)

    def _get_all_true_triples(self):
        all_true = defaultdict(set)
        all_data = np.concatenate([self.data.train, self.data.valid, self.data.test], axis=0)
        for h, r, t in all_data:
            all_true[(h, r)].add(t)
            all_true[(t, r + self.num_rel)].add(h)
        return all_true

    def evaluate_filtered(self, triples, batch_size=None):
            if batch_size is None:
                batch_size = self.config.get('eval_batch_size', 16)
                
            self.model.eval()
            ranks = []
            with torch.no_grad():
                for i in range(0, len(triples), batch_size):
                    batch = triples[i:i+batch_size]
                    h_cpu = batch[:, 0]
                    r_cpu = batch[:, 1]
                    t_correct = batch[:, 2]

                    h = torch.tensor(h_cpu, dtype=torch.long, device=self.device)
                    r = torch.tensor(r_cpu, dtype=torch.long, device=self.device)

                    logits = self.model(h, r)
                    
                    mask_rows = []
                    mask_cols = []
                    for j in range(len(batch)):
                        h_idx, r_idx, t_idx = int(h_cpu[j]), int(r_cpu[j]), int(t_correct[j])
                        others = list(self.all_true[(h_idx, r_idx)] - {t_idx})
                        if others:
                            mask_rows.extend([j] * len(others))
                            mask_cols.extend(others)
                    
                    if mask_rows:
                        logits[mask_rows, mask_cols] = -1e10

                    pos_scores = logits[torch.arange(len(batch)), t_correct].unsqueeze(1)
                    rank = torch.sum(logits > pos_scores, dim=1) + 1
                    ranks.extend(rank.cpu().tolist())

            ranks = np.array(ranks)
            return np.mean(1.0 / ranks), np.mean(ranks <= 10)

    def train(self, verbose=True):
            best_mrr = 0.0
            scaler = torch.amp.GradScaler('cuda')
            
            # 新增：记录整个训练过程的数据
            history = {
                'epoch': [],
                'train_loss': [],
                'valid_mrr': [],
                'valid_h10': [],
                'train_time': [],
                'eval_time': []
            }
            
            for epoch in range(self.config['epoch']):
                epoch_start_time = time.time() # 记录 Epoch 开始时间
                self.model.train()
                
                perm = torch.randperm(self.train_data.size(0), device=self.device)
                self.train_data = self.train_data[perm]
                
                total_loss = 0
                
                iterator = range(0, len(self.train_data), self.config['batch'])
                if verbose:
                    iterator = tqdm(iterator, desc=f"Epoch {epoch+1:03d}/{self.config['epoch']}", leave=False)
                    
                for i in iterator:
                    batch = self.train_data[i : i + self.config['batch']]
                    h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

                    self.optimizer.zero_grad()
                    
                    with torch.amp.autocast('cuda'):
                        logits_f = self.model(h, r)
                        loss_f = self.model.loss(logits_f, t)
                        
                        logits_b = self.model(t, r + self.num_rel)
                        loss_b = self.model.loss(logits_b, h)
                        
                        loss = (loss_f + loss_b) / 2

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    loss_val = loss.item()
                    total_loss += loss_val
                    
                    if verbose:
                        if isinstance(iterator, tqdm):
                            iterator.set_postfix(loss=f"{loss_val:.3f}")

                self.scheduler.step()
                train_time = time.time() - epoch_start_time # 计算训练耗时

                num_batches = len(range(len(self.train_data), self.config['batch']))
                avg_loss = total_loss / num_batches

                # 验证逻辑及时间记录
                eval_freq = self.config.get('eval_freq', 5)
                eval_time = 0.0
                mrr, h10 = None, None # 非评估 Epoch 设为 None
                
                if (epoch + 1) % eval_freq == 0:
                    eval_start_time = time.time()
                    mrr, h10 = self.evaluate_filtered(self.data.valid)
                    eval_time = time.time() - eval_start_time # 计算评估耗时
                    
                    if mrr > best_mrr:
                        best_mrr = mrr
                    if verbose:
                        print(f"Epoch {epoch+1:03d} | AvgLoss {avg_loss:.1f} | Valid MRR {mrr:.4f} | Hits@10 {h10:.4f} | TrainT {train_time:.1f}s | EvalT {eval_time:.1f}s")

                # 将当前 Epoch 的数据记录到 history
                history['epoch'].append(epoch + 1)
                history['train_loss'].append(avg_loss)
                history['train_time'].append(train_time)
                history['eval_time'].append(eval_time)
                history['valid_mrr'].append(mrr)
                history['valid_h10'].append(h10)

            # 最终测试集评估
            test_mrr, test_h10 = self.evaluate_filtered(self.data.test)
            if verbose:
                print(f"\n[Final Test] MRR: {test_mrr:.4f} | Hits@10: {test_h10:.4f}")
            
            return test_mrr, test_h10, history