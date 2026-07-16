import numpy as np
import os
import sys
sys.path.append('/content/FairDiffRec/CDiff4Rec')
import data_utils

def convert_to_recbole_format(dataset_name, data_dir):
    print(f"Converting dataset {dataset_name} to RecBole format...")
    
    train_path = os.path.join(data_dir, 'train_list.npy')
    valid_path = os.path.join(data_dir, 'valid_list.npy')
    test_path  = os.path.join(data_dir, 'test_list.npy')

    train_mat, valid_mat, test_mat, n_users, n_items = data_utils.data_load(train_path, valid_path, test_path)

    def save_inter(matrix, filename):
        with open(filename, 'w') as f:
            f.write("user_id:token\titem_id:token\trating:float\n")
            users, items = matrix.nonzero()
            for u, i in zip(users, items):
                f.write(f"{u + 1}\t{i + 1}\t1.0\n")

    save_inter(train_mat, os.path.join(data_dir, f'{dataset_name}.train.inter'))
    save_inter(valid_mat, os.path.join(data_dir, f'{dataset_name}.valid.inter'))
    save_inter(test_mat, os.path.join(data_dir, f'{dataset_name}.test.inter'))
    
    print(f"Conversion of {dataset_name} completed!\n")

if __name__ == '__main__':
    convert_to_recbole_format(dataset_name='ml-1m', data_dir='/content/FairDiffRec/datasets/ml-1m')
    convert_to_recbole_format(dataset_name='foursquare_tky', data_dir='/content/FairDiffRec/datasets/foursquare_tky')
    convert_to_recbole_format(dataset_name='books', data_dir='/content/FairDiffRec/datasets/books')
    convert_to_recbole_format(dataset_name='lastfm', data_dir='/content/FairDiffRec/datasets/lastfm')
