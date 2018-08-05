import numpy as np
import pandas as pd
import os

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event
from multiprocessing import Pool

from sklearn.neighbors import KDTree



cnt = 0
for event_id, hits, cells, particles, truth in load_dataset('../storage/track_ml_data/train_5.zip'):
    if cnt == 1:
        break
    cnt += 1


from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN

class Clusterer(object):
    def __init__(self,rz_scales=[0.65, 0.965, 1.528]):                        
        self.rz_scales=rz_scales
    
    def _eliminate_outliers(self,labels,M):
        norms=np.zeros((len(labels)),np.float32)
        indices=np.zeros((len(labels)),np.float32)
        for i, cluster in tqdm(enumerate(labels),total=len(labels)):
            if cluster == 0:
                continue
            index = np.argwhere(self.clusters==cluster)
            index = np.reshape(index,(index.shape[0]))
            indices[i] = len(index)
            x = M[index]
            norms[i] = self._test_quadric(x)
        threshold1 = np.percentile(norms,90)*5
        threshold2 = 25
        threshold3 = 6
        for i, cluster in enumerate(labels):
            if norms[i] > threshold1 or indices[i] > threshold2 or indices[i] < threshold3:
                self.clusters[self.clusters==cluster]=0   
    def _test_quadric(self,x):
        if x.size == 0 or len(x.shape)<2:
            return 0
        xm = np.mean(x,axis=0)
        x = x - xm
        Z = np.zeros((x.shape[0],10), np.float32)
        Z[:,0] = x[:,0]**2
        Z[:,1] = 2*x[:,0]*x[:,1]
        Z[:,2] = 2*x[:,0]*x[:,2]
        Z[:,3] = 2*x[:,0]
        Z[:,4] = x[:,1]**2
        Z[:,5] = 2*x[:,1]*x[:,2]
        Z[:,6] = 2*x[:,1]
        Z[:,7] = x[:,2]**2
        Z[:,8] = 2*x[:,2]
        Z[:,9] = 1
        v, s, t = np.linalg.svd(Z,full_matrices=False)        
        smallest_index = np.argmin(np.array(s))
        T = np.array(t)
        T = T[smallest_index,:]        
        norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
        return norm

    def _preprocess(self, hits):
        
        x = hits.x.values
        y = hits.y.values
        z = hits.z.values

        r = np.sqrt(x**2 + y**2 + z**2)
        hits['x2'] = x/r
        hits['y2'] = y/r

        r = np.sqrt(x**2 + y**2)
        hits['z2'] = z/r

        ss = StandardScaler()
        X = ss.fit_transform(hits[['x2', 'y2', 'z2']].values)
        for i, rz_scale in enumerate(self.rz_scales):
            X[:,i] = X[:,i] * rz_scale
       
        return X
    
    
    
    def find_labels(self, params):
        w1, w2, w3, w4, w5, w6, w7, epsilon = 2.7474448671796874,1.3649721713529086,0.7034918842926337,\
                                0.0005549122352940002,0.023096034747190672,0.04619756315527515,\
                                0.2437077420144654,0.009750302717746615
        
        hits, dz, z_shift, unroll_type = params

        hits['z'] = hits['z'] - z_shift
        hits['r'] = np.sqrt(hits['x'].values ** 2 + hits['y'].values ** 2 + hits['z'].values ** 2)
        hits['rt'] = np.sqrt(hits['x'].values ** 2 + hits['y'].values ** 2)
        hits['a0'] = np.arctan2(hits['y'].values, hits['x'].values)
        hits['z1'] = hits['z'].values / hits['rt'].values
        hits['z2'] = hits['z'].values / hits['r'].values
        hits['s1'] = hits['hit_id']
        hits['N1'] = 1
        hits['z1'] = hits['z'].values / hits['rt'].values
        hits['z2'] = hits['z'].values / hits['r'].values
        hits['x1'] = hits['x'].values / hits['y'].values
        hits['x2'] = hits['x'].values / hits['r'].values
        hits['x3'] = hits['y'].values / hits['r'].values
        hits['x4'] = hits['rt'].values / hits['r'].values
        
        if unroll_type == 0:
            hits['a1'] = hits['a0'].values + np.nan_to_num(np.arccos(dz*hits['rt'].values))
        if unroll_type == 1:
            hits['a1'] = hits['a0'].values + dz*hits['rt'].values
        if unroll_type == 2:
            hits['a1'] = hits['a0'].values + dz*hits['z'].values
        if unroll_type == 3:
            hits['a1'] = hits['a0'].values + dz * (hits['rt'].values + 0.000005 * hits['rt'].values ** 2)
        #hits['a1'] = hits['a0'].values + np.nan_to_num(np.arccos(dz*hits['rt'].values))

        hits['sina1'] = np.sin(hits['a1'].values)
        hits['cosa1'] = np.cos(hits['a1'].values)
        ss = StandardScaler()
        hits = ss.fit_transform(hits[['sina1', 'cosa1', 'z1', 'z2','x1','x2','x3','x4']].values)
        cx = np.array([w1, w1, w2, w3, w4, w5, w6, w7])
        hits = np.multiply(hits, cx)
        clusters = DBSCAN(eps=0.009750302717746615, min_samples=1, metric="euclidean", n_jobs=32).fit(hits).labels_
        return clusters    
    
    
    
    
    def _init(self, hits, Niter):
        
#         w1, w2, w3, w4, w5, w6, w7, epsilon = 2.7474448671796874,1.3649721713529086,0.7034918842926337,\
#                                         0.0005549122352940002,0.023096034747190672,0.04619756315527515,\
#                                         0.2437077420144654,0.009750302717746615
        
        
        

        
        
        params = []
        for i in range(Niter):
            
            unroll_type = np.random.randint(0,4)
            if unroll_type == 0:
                dz = np.random.normal(0.0, 0.00035)
            elif unroll_type == 1:
                dz = np.random.normal(0.0, 0.00065)
            elif unroll_type == 2:
                dz = np.random.normal(0.0, 0.00085)
            elif unroll_type == 3:
                dz = np.random.normal(0.0, 0.001)

            
            
            
            #dz = 1 / 1000 * (ii / 2) / 180 * np.pi
            #dz = np.random.normal(0.0, 0.001)
            #dz = np.random.normal(0.0, 0.00035)
 
            z_shift = np.random.normal(0.0, 4.5)


            params.append((hits, dz, z_shift, unroll_type))
            
        pool = Pool(processes=8)
        result = []
        for i in tqdm(pool.imap(self.find_labels, params)):
            result += [i]
        pool.close()
                
            
        return np.array(result)
    
    def predict(self, hits, Niter):         
        result  = self._init(hits, Niter)
#         X = self._preprocess(hits) 
#         cl = hdbscan.HDBSCAN(min_samples=1,min_cluster_size=7,
#                              metric='braycurtis',cluster_selection_method='leaf',algorithm='best', leaf_size=50)
#         labels = np.unique(self.clusters)
#         self._eliminate_outliers(labels,X)          
#         max_len = np.max(self.clusters)
#         mask = self.clusters == 0
#         self.clusters[mask] = cl.fit_predict(X[mask])+max_len
        return result





model = Clusterer()
result = model.predict(hits, 60000)


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


def merge(cl1, cl2, min_cnt): # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1':cl1,'s2':cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max()
    cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<20) &  (d['N2'].values>min_cnt))
    #cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<20) )

    s1 = d['s1'].values 
    s1[cond] = d['s2'].values[cond]+maxs1 
    return s1


#result = np.array(second)
labels = range(result.shape[1])

for k in [0]:
    for i in range(len(result[:])):
        labels = merge(labels, result[i], k)

    submission = create_one_event_submission(0, hits['hit_id'].values, labels)
    print(score_event(truth, submission))
    
    
np.save('predicts/luis_60k', result)    
    
 












