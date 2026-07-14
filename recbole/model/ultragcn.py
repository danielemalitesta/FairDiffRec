import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.utils import InputType

class UltraGCN(GeneralRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(UltraGCN, self).__init__(config, dataset)

        self.embedding_size = config['embedding_size'] if 'embedding_size' in config else 64
        self.negative_weight = config['negative_weight'] if 'negative_weight' in config else 15.0
        self.w1 = config['w1'] if 'w1' in config else 1.0
        self.w2 = config['w2'] if 'w2' in config else 1.0
        self.w3 = config['w3'] if 'w3' in config else 1.0
        self.w4 = config['w4'] if 'w4' in config else 1.0
        self.lambda_ = config['ILoss_lambda'] if 'ILoss_lambda' in config else 1e-4
        self.ii_neighbor_num = config['K'] if 'K' in config else 10 

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        
        nn.init.normal_(self.user_embedding.weight, std=1e-4)
        nn.init.normal_(self.item_embedding.weight, std=1e-4)

        start_time = time.time()
        self._compute_graph_weights(dataset)

    def _compute_graph_weights(self, dataset):
        inter_M = dataset.inter_matrix(form='csr').astype(np.float32)

        row_sum = np.array(inter_M.sum(axis=1)).squeeze()
        col_sum = np.array(inter_M.sum(axis=0)).squeeze()
        
        self.user_degree = torch.tensor(row_sum, dtype=torch.float32, device=self.device)
        self.item_degree = torch.tensor(col_sum, dtype=torch.float32, device=self.device)

        G = inter_M.T.dot(inter_M)
        g_i = np.array(G.sum(axis=1)).squeeze() + 1e-8
        
        G_no_diag = G.copy().tolil()
        G_no_diag.setdiag(0)
        G_no_diag = G_no_diag.tocsr()
        g_i_minus_diag = np.array(G_no_diag.sum(axis=1)).squeeze() + 1e-8

        topk_idx = []
        topk_weights = []

        for i in range(self.n_items):
            row = G_no_diag.getrow(i)
            if row.nnz == 0:
                topk_idx.append(np.zeros(self.ii_neighbor_num, dtype=np.int64))
                topk_weights.append(np.zeros(self.ii_neighbor_num, dtype=np.float32))
                continue

            cols = row.indices
            vals = row.data
            omega_vals = (vals / g_i_minus_diag[i]) * np.sqrt(g_i[i] / g_i[cols])

            if len(cols) >= self.ii_neighbor_num:
                top_k_indices = np.argpartition(omega_vals, -self.ii_neighbor_num)[-self.ii_neighbor_num:]
                sorted_top_k = top_k_indices[np.argsort(-omega_vals[top_k_indices])]
                topk_idx.append(cols[sorted_top_k])
                topk_weights.append(omega_vals[sorted_top_k])
            else:
                pad_len = self.ii_neighbor_num - len(cols)
                sorted_indices = np.argsort(-omega_vals)
                padded_idx = np.pad(cols[sorted_indices], (0, pad_len), 'constant', constant_values=0)
                padded_weights = np.pad(omega_vals[sorted_indices], (0, pad_len), 'constant', constant_values=0.0)
                topk_idx.append(padded_idx)
                topk_weights.append(padded_weights)

        self.topk_items = torch.tensor(np.array(topk_idx), dtype=torch.long, device=self.device)
        self.omega_weights = torch.tensor(np.array(topk_weights), dtype=torch.float32, device=self.device)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e = self.user_embedding(user)
        pos_e = self.item_embedding(pos_item)
        neg_e = self.item_embedding(neg_item)

        pos_scores = torch.sum(user_e * pos_e, dim=1)
        neg_scores = torch.sum(user_e * neg_e, dim=1)

        d_u = self.user_degree[user]
        d_i = self.item_degree[pos_item]
        d_j = self.item_degree[neg_item]

        beta_pos = (1.0 / (d_u + 1e-8)) * torch.sqrt((d_u + 1.0) / (d_i + 1.0))
        beta_neg = (1.0 / (d_u + 1e-8)) * torch.sqrt((d_u + 1.0) / (d_j + 1.0))

        pos_bce = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores), reduction='none')
        neg_bce = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores), reduction='none')

        loss_pos = (self.w1 + self.w3 * beta_pos) * pos_bce
        loss_neg = (self.w2 + self.w4 * beta_neg) * neg_bce * self.negative_weight

        ii_neighbors = self.topk_items[pos_item]
        ii_weights = self.omega_weights[pos_item]

        neighbor_e = self.item_embedding(ii_neighbors)
        ii_scores = torch.sum(user_e.unsqueeze(1) * neighbor_e, dim=2) 
        
        ii_bce = F.binary_cross_entropy_with_logits(ii_scores, torch.ones_like(ii_scores), reduction='none')
        loss_ii = torch.sum(ii_weights * ii_bce, dim=1) * self.lambda_

        loss = (loss_pos + loss_neg + loss_ii).mean()
        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e = self.user_embedding(user)
        item_e = self.item_embedding(item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.user_embedding(user)
        all_item_e = self.item_embedding.weight
        return torch.matmul(user_e, all_item_e.transpose(0, 1))
