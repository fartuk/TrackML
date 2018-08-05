
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import hdbscan
from scipy import stats
from tqdm import tqdm
from sklearn.cluster import DBSCAN
#from trackml.dataset import load_event, load_dataset
#from trackml.score import score_event
from IPython.display import clear_output
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from tqdm import tqdm
from multiprocessing import Pool


cnt = 0
for event_id, hits, cells, particles, truth in load_dataset('../storage/track_ml_data/train_5.zip'):
    if cnt == 1:
        break
    cnt += 1



c = [1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
min_samples_in_cluster = 1


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

def extract_good_hits(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    return tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]

def fast_score(good_hits_df):
    return good_hits_df.weight.sum()





class Clusterer(object):
    def __init__(self):                        
        self.abc = []
          
    def initialize(self,dfhits):
        self.cluster = range(len(dfhits))
        
        
 
    def find_labels(self, params):
        hits, dz, z_shift, unroll_type = params

        hits = preprocess_hits(hits, z_shift)
        
        
        
        if unroll_type == 0:
            hits['a1'] = hits['a0'].values + np.nan_to_num(np.arccos(dz*hits['rt'].values))
        if unroll_type == 1:
            hits['a1'] = hits['a0'].values + dz*hits['rt'].values
        if unroll_type == 2:
            hits['a1'] = hits['a0'].values + dz*hits['z'].values                
                        

        hits['sina1'] = np.sin(hits['a1'].values)
        hits['cosa1'] = np.cos(hits['a1'].values)

        ss = StandardScaler()
        hits = ss.fit_transform(hits[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']].values) 
        #dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
        hits = np.multiply(hits, c)
        new_cluster=DBSCAN(eps=0.0048,min_samples=1,metric='euclidean',n_jobs=8).fit(hits).labels_
        return new_cluster
        #merged_cluster = merge(merged_cluster, new_cluster)
        #result += [new_cluster]
#         if verbose == True:
#             sub = create_one_event_submission(0, hits, merged_cluster)
#             good_hits = extract_good_hits(truth, sub)
#             score_1 = fast_score(good_hits)
#             print('2r0_inverse:', ii*mm ,'. Score:', score_1)
                    
    
        
        
    def Hough_clustering(self,hits, epsilon, verbose=True): # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        #merged_cluster = self.cluster
        stepii = 0.000005
        count_ii = 0
        adaptive_eps_coefficient = 1
        
        result = []
        
        params = []
        for i in range(0, 60000):
            unroll_type = np.random.randint(0,3)
            if unroll_type == 0:
                dz = np.random.normal(0.0, 0.00035)
            elif unroll_type == 1:
                dz = np.random.normal(0.0, 0.00065)
            elif unroll_type == 2:
                dz = np.random.normal(0.0, 0.00085)

            #dz = 0
            z_shift = np.random.normal(0.0, 4.5)
            #z_shift = 0
            eps_new = epsilon + i*1*10**(-5)

            params.append((hits, dz, z_shift, unroll_type))
            
            
        
             
        pool = Pool(processes=8)
        result = []
        for i in tqdm(pool.imap(self.find_labels, params)):
            result += [i]
        pool.close()
                
            
        #self.cluster = merged_cluster
        return np.array(result)

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

def preprocess_hits(h,dz):
    h['z'] =  h['z'].values + dz
    h['r'] = np.sqrt(h['x'].values**2+h['y'].values**2+h['z'].values**2)
    h['rt'] = np.sqrt(h['x'].values**2+h['y'].values**2)
    h['a0'] = np.arctan2(h['y'].values,h['x'].values)
    h['zdivrt'] = h['z'].values/h['rt'].values
    h['zdivr'] = h['z'].values/h['r'].values
    h['xdivr'] = h['x'].values / h['r'].values
    h['ydivr'] = h['y'].values / h['r'].values
    return h






c = [1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
min_samples_in_cluster = 1

model = Clusterer()
#model.initialize(hits)

#hits_with_dz = preprocess_hits(hits, 0.055*i)

result = model.Hough_clustering(hits,epsilon=0.0048,verbose=True)





second = []
for k in range(100):
    #result = np.vstack([res0, res1, res2])
    np.random.shuffle(result)

    #result = res0
    labels = range(result.shape[1])

    for k in [0]:
        for i in range(len(result[:])):
            labels = merge(labels, result[i], k)

        submission = create_one_event_submission(0, hits['hit_id'].values, labels)
        print(score_event(truth, submission))
        
    second += [labels]
    
np.save('second', np.array(second))












