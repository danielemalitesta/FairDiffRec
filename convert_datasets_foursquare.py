import numpy as np

data = 'foursquare_tky'


train_origin = np.load(f'{data}/{data}.train', allow_pickle=True)
test_origin = np.load(f'{data}/{data}.test', allow_pickle=True)
valid_origin = np.load(f'{data}/{data}.validation', allow_pickle=True)


users = np.unique(train_origin['user_id']).tolist()
items = np.unique(train_origin['venue_id']).tolist()


users_map = {u: idx for idx, u in enumerate(users)}
items_map = {i: idx for idx, i in enumerate(items)}


train = np.array([[users_map[u] for u in train_origin['user_id'].tolist()],
                  [items_map[i] for i in train_origin['venue_id'].tolist()]])


user_list, item_list = [], []
for u, i in zip(test_origin['user_id'].tolist(), test_origin['venue_id'].tolist()):
    if i in items and u in users:
        user_list.append(users_map[u])
        item_list.append(items_map[i])

test = np.array([user_list, item_list])


user_list, item_list = [], []
for u, i in zip(valid_origin['user_id'].tolist(), valid_origin['venue_id'].tolist()):
    if i in items and u in users:
        user_list.append(users_map[u])
        item_list.append(items_map[i])

valid = np.array([user_list, item_list])


np.save(f'{data}/train_list.npy', train.transpose())
np.save(f'{data}/test_list.npy', test.transpose())
np.save(f'{data}/valid_list.npy', valid.transpose())


with open(f'{data}/users_map.tsv', 'w') as f: 
    for u, idx in users_map.items():
        f.write(f"{u}\t{idx}\n")

with open(f'{data}/items_map.tsv', 'w') as f: 
    for i, idx in items_map.items():
        f.write(f"{i}\t{idx}\n")

