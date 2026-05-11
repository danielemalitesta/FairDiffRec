"""
Train a diffusion model for recommendation
"""

import argparse
from ast import parse
import os
import time
import numpy as np
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
import evaluate_utils
import data_utils
from copy import deepcopy

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='yelp_clean', help='choose the dataset')
parser.add_argument('--data_path', type=str, default='./datasets/', help='load data path')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=400)
parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
parser.add_argument('--log_name', type=str, default='log', help='the log name')
parser.add_argument('--round', type=int, default=1, help='record the experiment')

# params for the model
parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

# params for diffusion
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

parser.add_argument('--regs', type=float, default=1e-3, help='l2 reg')
parser.add_argument('--lamd1', type=float, default=0.1, help='lamd1')
parser.add_argument('--lamd2', type=float, default=0.1, help='lamd2')

args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda:0" if args.cuda else "cpu")

print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

### DATA LOAD ###
train_path = args.data_path + args.dataset + '/train_list.npy'
valid_path = args.data_path + args.dataset + '/valid_list.npy'
test_path = args.data_path + args.dataset + '/test_list.npy'

train_data, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A))
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=4, worker_init_fn=worker_init_fn)
test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

if args.tst_w_val:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A) + torch.FloatTensor(valid_y_data.A))
    test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
mask_tv = train_data + valid_y_data

print('data ready.')


### Build Gaussian Diffusion ###
if args.mean_type == 'x0':
    mean_type = gd.ModelMeanType.START_X
elif args.mean_type == 'eps':
    mean_type = gd.ModelMeanType.EPSILON
else:
    raise ValueError("Unimplemented mean type %s" % args.mean_type)

diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
        args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

### Build MLP ###
out_dims = eval(args.dims) + [n_item]
in_dims = out_dims[::-1]
model = DNN(in_dims, out_dims, args.emb_size, args.lamd1, args.lamd2, time_type="cat", norm=args.norm).to(device)

optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
print("models ready.")

param_num = 0
mlp_num = sum([param.nelement() for param in model.parameters()])
diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
param_num = mlp_num + diff_num
print("Number of all parameters:", param_num)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def evaluate(data_loader, data_te, mask_his, topN, write=False, write_path=None):
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]

    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    
    if write and write_path:
        tot_users = 0
        with open(write_path, 'w') as f:
            with torch.no_grad():
                for batch_idx, batch in enumerate(data_loader):
                    his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                    batch = batch.to(device)
                    prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
                    prediction[his_data.nonzero()] = -np.inf

                    values, indices = torch.topk(prediction, topN[-1])
                    indices = indices.cpu().numpy().tolist()
                    predict_items.extend(indices)
                    
                    for user in range(batch.shape[0]):
                        current_values = values[user]
                        for idx, item in enumerate(indices[user]):
                            f.write(f'{tot_users}\t{item}\t{current_values[idx].item()}\n')
                        tot_users += 1
    else:
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                batch = batch.to(device)
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
                prediction[his_data.nonzero()] = -np.inf

                _, indices = torch.topk(prediction, topN[-1])
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)

    test_results = evaluate_utils.computeTopNAccuracy(target_items, predict_items, topN)

    return test_results

best_recall, best_epoch = -100, 0
best_test_result = None
best_model_state_dict = None

print("Start training...")
for epoch in range(1, args.epochs + 1):
    if epoch - best_epoch >= 50:
        print('-'*18)
        print('Exiting from training early')
        break

    model.train()
    start_time = time.time()

    batch_count = 0
    total_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        batch = batch.to(device)
        batch_count += 1
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, batch, args.reweight)
        loss = losses["loss"].mean()
        total_loss += loss
        loss.backward()
        optimizer.step()
    
    if epoch % 5 == 0:
        valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
        if args.tst_w_val:
            test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
        else:
            test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
        evaluate_utils.print_results(None, valid_results, test_results)
        
        if valid_results[1][1] > best_recall:
            best_recall, best_epoch = valid_results[1][1], epoch
            best_results = valid_results
            best_test_results = test_results
            # Saving the best state in memory
            best_model_state_dict = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    
    print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                        "%H: %M: %S", time.gmtime(time.time()-start_time)))
    print('---'*18)

print('==='*18)
print("End. Best Epoch {:03d} ".format(best_epoch))
evaluate_utils.print_results(None, best_results, best_test_results)   

#final save
if best_model_state_dict is not None:
    val_rec = best_results[1][1]
    test_rec = best_test_results[1][1]

    folder_name = f"_valid_recall_{val_rec:.4f}_test_recall_{test_rec:.4f}"
    save_dir_path = os.path.join('saved_models', folder_name)
    os.makedirs(save_dir_path, exist_ok=True)

    save_path = os.path.join(save_dir_path, "model.pth")
    torch.save(best_model_state_dict, save_path)

    model.load_state_dict(best_model_state_dict)
    model.to(device)
    final_tsv_path = os.path.join(save_dir_path, "best_recommendations.tsv")
    
    if args.tst_w_val:
        evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN), write=True, write_path=final_tsv_path)
    else:
        evaluate(test_loader, test_y_data, mask_tv, eval(args.topN), write=True, write_path=final_tsv_path)
    print(f"Recommendations saved in: {final_tsv_path}")

print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
