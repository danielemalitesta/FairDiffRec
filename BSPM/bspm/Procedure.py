import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
import time
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2

def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0, write=False, write_path=None, test_dict=None, mode='test'):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    
    testDict: dict = test_dict if test_dict is not None else dataset.testDict
    Recmodel: model.LightGCN
    adj_mat = dataset.UserItemNet.tolil()
    
    if(world.simple_model == 'lgn-ide'):
        lm = model.LGCN_IDE(adj_mat)
        lm.train()
    elif(world.simple_model == 'gf-cf'):
        lm = model.GF_CF(adj_mat)
        lm.train()
    elif(world.simple_model == 'bspm'):
        lm = model.BSPM(adj_mat, world.config)
        lm.train()
    elif(world.simple_model == 'bspm-torch'):
        lm = model.BSPM_TORCH(adj_mat, world.config)
        lm.train()
        
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
        
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
               
    users = list(testDict.keys())
    e_N = len(users)
    
    predicted_matrix = np.empty((e_N, dataset.m_items))
    f = open(write_path, 'w') if (write and write_path) else None
    tot_users = 0

    with torch.no_grad():
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long().to(world.device)
            
            if(world.simple_model in ['gf-cf','bspm']):
                rating = lm.getUsersRating(batch_users, world.dataset)
                rating = torch.from_numpy(rating).to(world.device)
            elif(world.simple_model == 'bspm-torch'):
                if not torch.is_tensor(adj_mat):
                    adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()
                batch_ratings = adj_mat[batch_users, :].to(world.device)
                rating = lm.getUsersRating(batch_ratings, world.dataset)
            else:
                rating = Recmodel.getUsersRating(batch_users_gpu)
            
            predicted_matrix[tot_users:tot_users + rating.shape[0], :] = rating.cpu().numpy()
            
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
                
                if mode == 'test' and hasattr(dataset, 'validDict'):
                    uid = batch_users[range_i]
                    if uid in dataset.validDict:
                        val_items = dataset.validDict[uid]
                        exclude_index.extend([range_i] * len(val_items))
                        exclude_items.extend(val_items)
                        
            rating[exclude_index, exclude_items] = -(1<<10)
            
            values, rating_K = torch.topk(rating, k=max_K)
            indices = rating_K.cpu().numpy().tolist()
            
            if f is not None:
                values_np = values.cpu().numpy()
                for u_idx, original_user_id in enumerate(batch_users):
                    current_values = values_np[u_idx]
                    for idx, item in enumerate(indices[u_idx]):
                        f.write(f'{original_user_id}\t{item}\t{current_values[idx].item()}\n')
            
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
            tot_users += rating.shape[0]
            
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
                
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        
        if world.tensorboard:
            w.add_scalars(f'{mode.capitalize()}/Recall@{world.topks}', {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'{mode.capitalize()}/Precision@{world.topks}', {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'{mode.capitalize()}/NDCG@{world.topks}', {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            
        if multicore == 1:
            pool.close()
            
    if f is not None:
        f.close()
        
    print(f"[{mode.upper()}] results: {results}")
    return results, predicted_matrix

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
