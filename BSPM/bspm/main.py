import world
import utils
from world import cprint
import torch
import numpy as np
import scipy.sparse as sp
import copy
import os
from torch.utils.tensorboard import SummaryWriter
import time
import Procedure
from os.path import join

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:

    if world.simple_model != 'none':
        epoch = 0
        cprint(f"[RUNNING NON-PARAMETRIC MODEL: {world.simple_model}]")
        
        cprint("[EVALUATING VALIDATION]")
        valid_results, _ = Procedure.Test(
            dataset, Recmodel, epoch, w, world.config['multicore'], 
            test_dict=dataset.validDict, mode='valid'
        )
        val_rec = valid_results['recall'][0]
        
        cprint("[EVALUATING TEST]")
        test_results, predicted_matrix = Procedure.Test(
            dataset, Recmodel, epoch, w, world.config['multicore'], 
            test_dict=dataset.testDict, mode='test'
        )
        test_rec = test_results['recall'][0]
        
        folder_name = f"_valid_recall_{val_rec:.4f}_test_recall_{test_rec:.4f}"
        save_dir_path = os.path.join('saved_models', folder_name)
        os.makedirs(save_dir_path, exist_ok=True)
        
        final_tsv_path = os.path.join(save_dir_path, "best_recommendations.tsv")
        Procedure.Test(
            dataset, Recmodel, epoch, w, world.config['multicore'], 
            write=True, write_path=final_tsv_path, test_dict=dataset.testDict, mode='test'
        )
        print(f"Recommendations saved in: {final_tsv_path}")
        
        dataset_name_clean = world.dataset.replace("/", "")
        base_filename = os.path.join(save_dir_path, f'matrices_{dataset_name_clean}')
        
        sp.save_npz(f'{base_filename}_original.npz', dataset.UserItemNet)
        np.save(f'{base_filename}_predicted.npy', predicted_matrix)
        print(f"Matrices saved in:\n- {base_filename}_original.npz\n- {base_filename}_predicted.npy")


    else:
        best_recall = -100
        best_epoch = 0
        patience = 5  
        patience_counter = 0
        best_results = None
        best_test_results = None
        best_model_state_dict = None

        for epoch in range(world.TRAIN_epochs):

            if epoch % 10 == 0 and epoch > 0:
                cprint("[EVALUATING VALIDATION]")
                valid_results, _ = Procedure.Test(
                    dataset, Recmodel, epoch, w, world.config['multicore'], 
                    test_dict=dataset.validDict, mode='valid'
                )
                
                cprint("[EVALUATING TEST]")
                test_results, _ = Procedure.Test(
                    dataset, Recmodel, epoch, w, world.config['multicore'], 
                    test_dict=dataset.testDict, mode='test'
                )
                
                current_recall = valid_results['recall'][0]
                

                if current_recall > best_recall:
                    best_recall = current_recall
                    best_epoch = epoch
                    best_results = valid_results
                    best_test_results = test_results
                    patience_counter = 0
                    

                    best_model_state_dict = copy.deepcopy({k: v.cpu() for k, v in Recmodel.state_dict().items()})
                    cprint(f" >>> Miglior modello aggiornato! Val Recall: {best_recall:.5f} | Test Recall: {test_results['recall'][0]:.5f} <<<")
                else:
                    patience_counter += 1
                    print(f" Nessun miglioramento. Pazienza: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    cprint(f" [EARLY STOPPING] L'addestramento si interrompe all'epoca {epoch}.")
                    break

            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k, w=w)
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            
        print('==='*18)
        print("End. Best Epoch {:03d} ".format(best_epoch))
        

        if best_model_state_dict is not None:
            val_rec = best_results['recall'][0]
            test_rec = best_test_results['recall'][0]

            folder_name = f"_valid_recall_{val_rec:.4f}_test_recall_{test_rec:.4f}"
            save_dir_path = os.path.join('saved_models', folder_name)
            os.makedirs(save_dir_path, exist_ok=True)

            save_path = os.path.join(save_dir_path, "model.pth")
            torch.save(best_model_state_dict, save_path)
            print(f"Model state dict saved to: {save_path}")
            
            Recmodel.load_state_dict(best_model_state_dict)
            Recmodel = Recmodel.to(world.device)
            
            final_tsv_path = os.path.join(save_dir_path, "best_recommendations.tsv")
            _, predicted_matrix = Procedure.Test(
                dataset, Recmodel, best_epoch, w, world.config['multicore'], 
                write=True, write_path=final_tsv_path, test_dict=dataset.testDict, mode='test'
            )
            print(f"Recommendations saved in: {final_tsv_path}")

            dataset_name_clean = world.dataset.replace("/", "")
            base_filename = os.path.join(save_dir_path, f'matrices_{dataset_name_clean}')
            
            sp.save_npz(f'{base_filename}_original.npz', dataset.UserItemNet)
            np.save(f'{base_filename}_predicted.npy', predicted_matrix)
            print(f"Matrices saved in:\n- {base_filename}_original.npz\n- {base_filename}_predicted.npy")

finally:
    if world.tensorboard and w is not None:
        w.close()
