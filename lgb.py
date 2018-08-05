from numba import jit
import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from tqdm import tqdm
from multiprocessing import Pool
from sklearn.neighbors import KDTree

@jit(nopython=True)
def clusters_to_rows_cols(clusters):
    sort_inds = np.argsort(clusters)
        
    rows = []
    cols = []
    
    cluster_start = 0
    cluster_end = -1
    
    for i in range(len(sort_inds)):
        value = clusters[sort_inds[i]]
        
        if i==len(sort_inds)-1 or value != clusters[sort_inds[i+1]]:
            cluster_end = i+1
            
            cluster_inds = sort_inds[cluster_start:cluster_end]
            
            for k in range(len(cluster_inds)):
                rows.append(cluster_inds[k])
                cols.append(cluster_inds[k])
                
                for l in range(k+1, len(cluster_inds)):
                    rows.append(cluster_inds[k])
                    cols.append(cluster_inds[l])
                    
                    rows.append(cluster_inds[l])
                    cols.append(cluster_inds[k])
                    
            cluster_start = i+1
            previous_value = value
    return np.array(rows), np.array(cols)            
    #return rows, cols
            

def clusters_to_sparse(clusters):
    rows, cols = clusters_to_rows_cols(clusters)
    
    data = np.array([1], dtype = np.uint8)
    data = np.lib.stride_tricks.as_strided(data, shape = [len(rows)], strides=[0], writeable=False)
    
    
    return csr_matrix((data, (rows, cols)), dtype = np.uint8)
    





ev = []
hi = []
ce = []
pa = []
tr = []
cnt = 0
for event_id, hits, cells, particles, truth in load_dataset('../storage/track_ml_data/train_5.zip'):
    if cnt == 6:
        break
    cnt += 1
    hi += [hits]
    tr += [truth]
    



def get_feat(i):
    
    result = np.load('predicts/luis_{}.npy'.format(i))
    
    a, val = np.unique(tr[i]['particle_id'].values, return_inverse=True)
    gt_M = clusters_to_sparse(val)
    
    matrices = []
    for k in tqdm(range(3000)):
        matrices += [clusters_to_sparse(result[k])]
    
    mean_M = csr_matrix((result.shape[1], result.shape[1]), dtype=np.int8)
    for m in tqdm(matrices):
        mean_M += m
    mean_M = mean_M / 3000
    
    idxs_1, idxs_2 = mean_M.nonzero()


    
    std_M = csr_matrix((result.shape[1], result.shape[1]), dtype=np.float16)
    for m in tqdm(matrices):
        std_M += (m-mean_M).power(2)
        std_M = std_M.sqrt() / 3000
    
    tmp = []
    for m in tqdm(matrices):
        tmp += [np.squeeze(np.asarray(m.sum(axis=0)))]
    tmp = np.array(tmp)
    
    
    X = pd.DataFrame()
    X['idxs_1'] = idxs_1
    X['idxs_2'] = idxs_2
    X['event'] = i
    X['mean'] = np.squeeze(np.asarray(mean_M[(idxs_1, idxs_2)]))
    X['std'] = np.squeeze(np.asarray(std_M[(idxs_1, idxs_2)]))
    
#     X['mean_len_1'] = np.max([tmp.mean(axis=0)[idxs_1], tmp.mean(axis=0)[idxs_2]], axis=0)
#     X['mean_len_2'] = np.min([tmp.mean(axis=0)[idxs_1], tmp.mean(axis=0)[idxs_2]], axis=0)

    X['mean_len_1'] = tmp.mean(axis=0)[idxs_1]
    X['mean_len_2'] = tmp.mean(axis=0)[idxs_2]

    X['std_len_1'] = tmp.std(axis=0)[idxs_1]
    X['std_len_2'] = tmp.std(axis=0)[idxs_2]
    
    X['max_len_1'] = tmp.max(axis=0)[idxs_1]
    X['max_len_2'] = tmp.max(axis=0)[idxs_2]
    
    X['min_len_1'] = tmp.min(axis=0)[idxs_1]
    X['min_len_2'] = tmp.min(axis=0)[idxs_2]
    
    
    
    tmp = np.squeeze(np.asarray((mean_M > 0).sum(axis=0)))
    
    
    X['tot_len_1'] = tmp[idxs_1]
    X['tot_len_2'] = tmp[idxs_2]
    #X['tot_len_common'] = np.squeeze(np.asarray(csr_matrix((mean_M > 0).multiply((mean_M > 0).sum(axis=0)))[(idxs_1, idxs_2)]))
    
    sh = np.squeeze(np.asarray(csr_matrix(m.multiply(m.sum(axis=0)))[(idxs_1, idxs_2)])).shape[0]
    tmp_quad = np.zeros(sh)
    tmp_sum = np.zeros(sh)
    tmp_max = np.zeros(sh)
    tmp_min = np.ones(sh) * 3000
    for m in tqdm(matrices):
        a = np.squeeze(np.asarray(csr_matrix(m.multiply(m.sum(axis=0)))[(idxs_1, idxs_2)]))# ** 2
        tmp_sum += a
        tmp_quad += a ** 2
        tmp_max = np.max([tmp_max, a], axis=0)
        tmp_min = np.min([tmp_min, a], axis=0)

    #tmp = np.array(tmp)
    X['mean_common_len'] = tmp_sum / 3000
    X['std_common_len'] = (tmp_quad - tmp_sum ** 2) / 3000
    X['max_common_len'] = tmp_max
    X['min_common_len'] = tmp_min

    
    

    
    
    X['target'] = np.squeeze(np.asarray(gt_M[mean_M.nonzero()]))

    return X, i

import lightgbm as lgbm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pickle
pool = Pool(processes=32)
X_train = pd.DataFrame()
for X, i in tqdm(pool.imap(get_feat, range(1000, 1032))):
    X.to_csv('data/tables/{}'.format(i), index=False)
    



    lgb = lgbm.sklearn.LGBMClassifier(n_jobs=1)
    lgb.fit(X.drop(['idxs_1','idxs_2', 'event', 'target'] ,axis=1), X['target'])

    with open('data/models/{}.pickle'.format(i), 'wb') as f:
        pickle.dump(lgb, f)  
        
pool.close()











