import argparse
import sys
import numpy as np
import torch
import os
from recbole.data.interaction import Interaction 

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from recbole.trainer import HyperTuning
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, get_model
from recbole.trainer import Trainer

GLOBAL_MODEL = ""
GLOBAL_DATASET = ""
METRIC_K = 20
EXPORT_K = 100

def get_base_config():
    return {
        'model': GLOBAL_MODEL,
        'dataset': GLOBAL_DATASET,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'load_col': {'inter': ['user_id', 'item_id']},
        
        'benchmark_filename': ['train', 'valid', 'test'],
        'eval_args': {
            'split': {'RS': [1, 0, 0]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        },
        
        'topk': [METRIC_K],
        'metrics': ['Recall', 'NDCG'],
        'valid_metric': f'Recall@{METRIC_K}',
        
        'eval_step': 5,
        'stopping_step': 6,
        
        'data_path': '/content/FairDiffRec/datasets/'
    }

def custom_objective_function(config_dict=None, config_file_list=None, saved=True):
    base_config = get_base_config()
    
    base_config['epochs'] = 200
    
    if config_dict:
        base_config.update(config_dict)

    config = Config(config_dict=base_config, config_file_list=config_file_list)
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model_class = get_model(config['model'])
    model = model_class(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, verbose=False, saved=True)
    
    test_result = trainer.evaluate(test_data, load_best_model=True)

    if hasattr(trainer, 'saved_model_file') and os.path.exists(trainer.saved_model_file):
        os.remove(trainer.saved_model_file)

    return {
        'best_valid_score': best_valid_score,
        'valid_score_bigger': config['valid_metric_bigger'],
        'best_valid_result': best_valid_result,
        'test_result': test_result,
        'model': config['model'] 
    }

def main():
    global GLOBAL_MODEL, GLOBAL_DATASET
    
    parser = argparse.ArgumentParser(description="Master Script: Grid Search + Final Training + TSV Export")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., NeuMF)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., lastfm)')
    parser.add_argument('--params_file', type=str, required=True, help='Path to parameters file (e.g., neumf.hyper)')
    parser.add_argument('--output_file', type=str, default='hyper_search_results.txt', help='Output file')
    
    args = parser.parse_args()
    GLOBAL_MODEL = args.model
    GLOBAL_DATASET = args.dataset
    
    # GRID SEARCH
    print(f"\n{'='*50}\nHYPERPARAMETER TUNING ({args.model} on {args.dataset})\n{'='*50}")
    
    hp = HyperTuning(
        objective_function=custom_objective_function,
        algo='exhaustive',
        params_file=args.params_file
    )
    
    hp.run()
    hp.export_result(output_file=args.output_file)
    
    print('\n--- Search Completed ---')
    print('Best parameters found: ', hp.best_params)
    
    yaml_filename = f"best_params_{args.model}_{args.dataset}.yaml"
    with open(yaml_filename, 'w') as f:
        for key, value in hp.best_params.items():
            f.write(f"{key}: {value}\n")

    # FINAL TRAINING
    print(f"\n{'='*50}\nFINAL TRAINING WITH BEST PARAMS\n{'='*50}")
    
    final_config_dict = get_base_config()
    final_config_dict.update(hp.best_params)
    
    final_config_dict['epochs'] = 1000
    
    config = Config(model=args.model, dataset=args.dataset, config_dict=final_config_dict)
    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model_class = get_model(config['model'])
    model = model_class(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True, saved=True)
    
    if hasattr(trainer, 'saved_model_file'):
        print(f"\n[INFO] Best Model Parameters (.pth) successfully saved to: {trainer.saved_model_file}")
    
    print(f"\n--- Final Evaluation on Test Set ---")
    test_result = trainer.evaluate(test_data, load_best_model=True)
    print(test_result)

    # RECOMMENDATIONS FILE GENERATION
    print(f"\n{'='*50}\nGENERATING TSV FILE (Top-{EXPORT_K})\n{'='*50}")
    model.eval()
    
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    output_filename = f'best_recommendations_{args.model}_{args.dataset}.tsv'
    
    eval_loader = test_data
    
    with open(output_filename, 'w') as f:
        for batched_data in eval_loader:
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(config['device'])
            
            with torch.no_grad():
                try:
                    scores = model.full_sort_predict(interaction)
                except NotImplementedError:
                    scores_list = []
                    all_items = torch.arange(dataset.item_num, device=config['device'])
                    
                    for i in range(len(interaction)):
                        input_dict = {}
                        input_dict[uid_field] = interaction[uid_field][i].repeat(dataset.item_num)
                        input_dict[iid_field] = all_items
                        
                        for k, v in interaction.interaction.items():
                            if k not in [uid_field, iid_field]:
                                input_dict[k] = v[i].repeat(dataset.item_num)
                                
                        input_inter = Interaction(input_dict).to(config['device'])
                        
                        user_scores = []
                        chunk_size = 50000 
                        for start_idx in range(0, dataset.item_num, chunk_size):
                            chunk_inter = input_inter[start_idx : start_idx + chunk_size]
                            chunk_scores = model.predict(chunk_inter)
                            user_scores.append(chunk_scores)
                            
                        scores_list.append(torch.cat(user_scores))
                        
                    scores = torch.stack(scores_list)
            
            scores = scores.view(-1, dataset.item_num)
            
            scores[:, 0] = -np.inf
            if history_index is not None:
                scores[history_index] = -np.inf
            
            topk_scores, topk_items = torch.topk(scores, EXPORT_K, dim=1)
            
            batch_users = interaction[uid_field].cpu().numpy()
            
            for row_idx, internal_user in enumerate(batch_users):
                user_token = dataset.field2id_token[uid_field][internal_user]
                user_id = int(user_token) - 1 
                
                for item_score, internal_item in zip(topk_scores[row_idx], topk_items[row_idx]):
                    internal_item = internal_item.item()
                    item_token = dataset.field2id_token[iid_field][internal_item]
                    item_id = int(item_token) - 1 
                    
                    f.write(f"{user_id}\t{item_id}\t{item_score.item()}\n")
                    
    print(f"\nSUCCESS! File '{output_filename}' saved.\n")

if __name__ == '__main__':
    main()
