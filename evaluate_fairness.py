import sys
import argparse
import pandas as pd
import numpy as np
import os

sys.path.append('/content/FairDiffRec/CDiff4Rec')
import data_utils

parser = argparse.ArgumentParser(description="Evaluate Fairness for Recommendation Systems")
parser.add_argument('--dataset', type=str, default='ml-1m')
parser.add_argument('--tsv_path', type=str, default='/content/FairDiffRec/best_recommendations.tsv')
# Thresholds for specific datasets
parser.add_argument('--age_threshold', type=int, default=30, help="Age threshold for books")
parser.add_argument('--activity_threshold', type=int, default=100, help="Interaction threshold for lastfm")
args = parser.parse_args()

K = 20
SH_RATIO = 0.2  
TSV_PATH = args.tsv_path
DATA_DIR = f'/content/FairDiffRec/datasets/{args.dataset}/'

recs = pd.read_csv(TSV_PATH, sep='\t', header=None, names=['user_id', 'item_id', 'score'])
train_path = DATA_DIR + 'train_list.npy'
valid_path = DATA_DIR + 'valid_list.npy'
test_path  = DATA_DIR + 'test_list.npy'
train_mat, valid_mat, test_mat, n_users, n_items = data_utils.data_load(train_path, valid_path, test_path)

# CONSUMER FAIRNESS
if args.dataset in ['ml-1m', 'foursquare_tky']:
    users_map = pd.read_csv(DATA_DIR + 'users_map.tsv', sep='\t', header=None, names=['org_id', 'int_id'], dtype=str)
    
    users_info_path = DATA_DIR + f'{args.dataset}.user'
      
    users_info = pd.read_csv(users_info_path, sep='\t', dtype=str)
    if len(users_info.columns) == 1:
        users_info = pd.read_csv(users_info_path, sep=';', dtype=str)
        
    col_user = [c for c in users_info.columns if 'user_id' in c][0]
    col_gender = [c for c in users_info.columns if 'gender' in c.lower()][0]

    user_gender = pd.merge(users_map, users_info, left_on='org_id', right_on=col_user)
    gender_dict = dict(zip(user_gender['int_id'].astype(int), user_gender[col_gender]))

    genders = list(set(gender_dict.values()))
    
    group1_name = str(genders[0])
    group2_name = str(genders[1])
    group1_users = [u for u, g in gender_dict.items() if g == genders[0]]
    group2_users = [u for u, g in gender_dict.items() if g == genders[1]]

elif args.dataset == 'books':
    users_map = pd.read_csv(DATA_DIR + 'users_map.tsv', sep='\t', header=None, names=['org_id', 'int_id'], dtype=str)
    users_info_path = DATA_DIR + 'users.csv'
        
    users_info = pd.read_csv(users_info_path, sep=';', dtype=str)
    
    col_user = [c for c in users_info.columns if 'user' in c.lower() or 'id' in c.lower()][0]
    col_age = [c for c in users_info.columns if 'age' in c.lower()][0]
    
    user_merged = pd.merge(users_map, users_info, left_on='org_id', right_on=col_user)
    user_merged[col_age] = pd.to_numeric(user_merged[col_age], errors='coerce')
    age_dict = dict(zip(user_merged['int_id'].astype(int), user_merged[col_age]))
    
    group1_name = f"Age <= {args.age_threshold}"
    group2_name = f"Age > {args.age_threshold}"
    
    group1_users = [u for u, a in age_dict.items() if pd.notna(a) and a <= args.age_threshold]
    group2_users = [u for u, a in age_dict.items() if pd.notna(a) and a > args.age_threshold]

elif args.dataset == 'lastfm':
    user_interactions = np.array(train_mat.sum(axis=1)).flatten()
    
    group1_name = f"Interactions <= {args.activity_threshold} (Low Active)"
    group2_name = f"Interactions > {args.activity_threshold} (High Active)"
    
    # Identify users based on their interaction count
    group1_users = np.where(user_interactions <= args.activity_threshold)[0].tolist()
    group2_users = np.where(user_interactions > args.activity_threshold)[0].tolist()

else:
    print(f"Error: Dataset '{args.dataset}' not recognized for Consumer Fairness.")
    sys.exit(1)

item_pop = np.array(train_mat.sum(axis=0)).flatten()
n_sh = int(n_items * SH_RATIO)
sh_items = np.argsort(item_pop)[::-1][:n_sh]

sh_mask = np.zeros(n_items, dtype=bool)
sh_mask[sh_items] = True
lt_mask = ~sh_mask

ndcg_list = np.zeros(n_users)
rec_list  = np.zeros(n_users)

recs['rank'] = recs.groupby('user_id').cumcount()
top_k_recs = recs[recs['rank'] < K]

for u in range(n_users):
    targets = test_mat[u].nonzero()[1]
    preds = top_k_recs[top_k_recs['user_id'] == u]['item_id'].values
    
    if len(targets) == 0 or len(preds) == 0:
        continue
        
    hits = np.isin(preds, targets)
    rec_list[u] = np.sum(hits) / len(targets)
    
    dcg = np.sum(hits / np.log2(np.arange(2, len(hits) + 2)))
    idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(targets), K) + 2)))
    ndcg_list[u] = dcg / idcg if idcg > 0 else 0.0

# Consumer Fairness Metrics
def get_delta(metric_array):
    g1_valid = [u for u in group1_users if len(test_mat[u].nonzero()[1]) > 0]
    g2_valid = [u for u in group2_users if len(test_mat[u].nonzero()[1]) > 0]
    
    m1 = np.mean(metric_array[g1_valid]) if len(g1_valid) > 0 else 0.0
    m2 = np.mean(metric_array[g2_valid]) if len(g2_valid) > 0 else 0.0
    return abs(m1 - m2)

delta_ndcg = get_delta(ndcg_list)
delta_rec = get_delta(rec_list)

# PROVIDER FAIRNESS
all_preds = top_k_recs['item_id'].values
raw_visibility = np.bincount(all_preds, minlength=n_items)
visibility_prob = raw_visibility / (n_users * K)

dist_sh = 1.0
dist_lt = (1.0 / SH_RATIO) - 1.0 

aplt = np.sum(visibility_prob[lt_mask])

exposure = np.zeros(n_items)
discount = 1.0 / np.log2(np.arange(2, K + 2))
exp_disc_sum = np.sum(discount)

for r in range(K):
    rank_items = top_k_recs[top_k_recs['rank'] == r]['item_id'].values
    counts = np.bincount(rank_items, minlength=n_items)
    exposure += counts * discount[r]
    
exposure = (exposure / exp_disc_sum) / n_users
sh_exp = np.sum(exposure[sh_mask]) / dist_sh
lt_exp = np.sum(exposure[lt_mask]) / dist_lt
delta_exp = abs(sh_exp - lt_exp)


# RESULTS
print("\n" + "="*50)
print(f" FAIRNESS RESULTS ({args.dataset.upper()} - K={K})")
print("="*50)
print(f"[Consumer Fairness]")
print(f" - Groups:         {group1_name} VS {group2_name}")
print(f" - Delta Recall:   {delta_rec:.5f}")
print(f" - Delta NDCG:     {delta_ndcg:.5f}")
print(f"\n[Provider Fairness]")
print(f" - Delta Exposure: {delta_exp:.5f}")
print(f" - APLT:           {aplt:.5f}")
print("="*50)
