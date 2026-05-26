import numpy as np
import scipy.sparse as sp
from scipy.stats import wasserstein_distance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--orig_path', type=str, required=True, help='Path of original matrix file')
parser.add_argument('--pred_path', type=str, required=True, help='Path of predicted matrix file')
args = parser.parse_args()

# Loading
train_data = sp.load_npz(args.orig_path)
predicted_matrix = np.load(args.pred_path)
print(f"Original data: {train_data.shape} | Predictions: {predicted_matrix.shape}")

# Building generated matrix
num_edges = int(np.sum(train_data))
flattened = predicted_matrix.flatten()
threshold = np.partition(flattened, -num_edges)[-num_edges]
generated_matrix = (predicted_matrix >= threshold).astype(int)


# degree metrics
original_user_degree = train_data.A.sum(1).astype(int)
generated_user_degree = generated_matrix.sum(1)

for u in np.argwhere(generated_user_degree == 0):
    u_idx = u[0]
    best_i = np.argmax(predicted_matrix[u_idx])
    generated_matrix[u_idx, best_i] = 1
    generated_user_degree[u_idx] = 1

wd_user = wasserstein_distance(original_user_degree, generated_user_degree)

original_item_degree = train_data.A.sum(0).astype(int)
generated_item_degree = generated_matrix.sum(0)

for i in np.argwhere(generated_item_degree == 0):
    i_idx = i[0]
    best_u = np.argmax(predicted_matrix[:, i_idx])
    generated_matrix[best_u, i_idx] = 1
    generated_item_degree[i_idx] = 1

wd_item = wasserstein_distance(original_item_degree, generated_item_degree)

print('\n=== Degree Metrics ===')
print(f"Wasserstein User: {wd_user:.2f} | Item: {wd_item:.2f}")
print(f'Orig. Users: {original_user_degree.mean():.2f}+-{original_user_degree.std():.2f} | Items: {original_item_degree.mean():.2f}+-{original_item_degree.std():.2f}')
print(f'Gen.  Users: {generated_user_degree.mean():.2f}+-{generated_user_degree.std():.2f} | Items: {generated_item_degree.mean():.2f}+-{generated_item_degree.std():.2f}')


# clustering metrics
def clust_coeff(inter, union):
    inter = np.asarray(inter)
    union = np.asarray(union)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = inter / union
        iou = np.nan_to_num(iou)
    np.fill_diagonal(iou, 0.0)
    num = iou.sum(1)
    denom = (inter > 0).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        res = num / denom
        res = np.nan_to_num(res)
    return res.flatten().tolist()

train_data_dense = np.asarray(train_data.A)

# Original
uu_o_inter = (train_data_dense @ train_data_dense.T).astype(int)
np.fill_diagonal(uu_o_inter, 0)
uu_o_union = original_user_degree[:, None] + original_user_degree[None, :] - uu_o_inter
np.fill_diagonal(uu_o_union, 0)

ii_o_inter = (train_data_dense.T @ train_data_dense).astype(int)
np.fill_diagonal(ii_o_inter, 0)
ii_o_union = original_item_degree[None, :] + original_item_degree[:, None] - ii_o_inter
np.fill_diagonal(ii_o_union, 0)

clust_user_orig = clust_coeff(uu_o_inter, uu_o_union)
clust_item_orig = clust_coeff(ii_o_inter, ii_o_union)

# Generated
uu_g_inter = (generated_matrix @ generated_matrix.T).astype(int)
np.fill_diagonal(uu_g_inter, 0)
uu_g_union = generated_user_degree[:, None] + generated_user_degree[None, :] - uu_g_inter
np.fill_diagonal(uu_g_union, 0)

ii_g_inter = (generated_matrix.T @ generated_matrix).astype(int)
np.fill_diagonal(ii_g_inter, 0)
ii_g_union = generated_item_degree[None, :] + generated_item_degree[:, None] - ii_g_inter
np.fill_diagonal(ii_g_union, 0)

clust_user_gen = clust_coeff(uu_g_inter, uu_g_union)
clust_item_gen = clust_coeff(ii_g_inter, ii_g_union)

wd_user_clust = wasserstein_distance(clust_user_orig, clust_user_gen)
wd_item_clust = wasserstein_distance(clust_item_orig, clust_item_gen)

print('\n=== Clustering Metrics ===')
print(f"Wasserstein Clust User: {wd_user_clust:.2f} | Item: {wd_item_clust:.2f}")
print(f'Orig. Users: {np.array(clust_user_orig).mean():.2f}+-{np.array(clust_user_orig).std():.2f} | Items: {np.array(clust_item_orig).mean():.2f}+-{np.array(clust_item_orig).std():.2f}')
print(f'Gen.  Users: {np.array(clust_user_gen).mean():.2f}+-{np.array(clust_user_gen).std():.2f} | Items: {np.array(clust_item_gen).mean():.2f}+-{np.array(clust_item_gen).std():.2f}')