import numpy as np
import os
from collections import defaultdict

dataset_dir = '/content/FairDiffRec/datasets/ml-1m/'

def convert_npy_to_txt(npy_filename, txt_filename):
    npy_path = os.path.join(dataset_dir, npy_filename)
    txt_path = os.path.join(dataset_dir, txt_filename)
    
    if not os.path.exists(npy_path):
        print(f"⚠️ File not found: {npy_path} (Skipping...)")
        return
        
    data = np.load(npy_path, allow_pickle=True)
    print(f"Parsing {npy_filename}...")
    
    with open(txt_path, 'w') as f:
        if hasattr(data, 'tocsr') or 'sparse' in str(type(data)):
            csr = data.tocsr() if hasattr(data, 'tocsr') else data
            for u in range(csr.shape[0]):
                items = csr[u].indices
                if len(items) > 0:
                    f.write(f"{u} {' '.join(map(str, items))}\n")
                    
        elif hasattr(data, 'ndim') and data.ndim == 2 and data.shape[1] > 3:
            for u in range(data.shape[0]):
                items = np.where(data[u] > 0)[0]
                if len(items) > 0:
                    f.write(f"{u} {' '.join(map(str, items))}\n")
                    
        elif hasattr(data, 'ndim') and data.ndim == 2 and data.shape[1] in [2, 3]:
            user_items = defaultdict(list)
            for row in data:
                u, i = int(row[0]), int(row[1])
                user_items[u].append(i)
            for u in sorted(user_items.keys()):
                f.write(f"{u} {' '.join(map(str, user_items[u]))}\n")
                
        else:
            for u, items in enumerate(data):
                if items is not None and len(items) > 0:
                    f.write(f"{u} {' '.join(map(str, items))}\n")
                    
    print(f"✅ Successfully created: {txt_filename}")

convert_npy_to_txt('train_list.npy', 'train.txt')
convert_npy_to_txt('valid_list.npy', 'valid.txt')
convert_npy_to_txt('test_list.npy', 'test.txt')
