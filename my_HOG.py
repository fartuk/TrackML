
# coding: utf-8

# In[1]:



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
from sklearn.neighbors import KDTree


# In[3]:


cnt = 0
for event_id, hits, cells, particles, truth in load_dataset('../storage/track_ml_data/train_5.zip'):
    if cnt == 1:
        break
    cnt += 1


# In[77]:





# In[102]:


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
        w1, w2, w3, w4, w5, w6, w7, epsilon = 2.7474448671796874,1.3649721713529086,0.7034918842926337,                                0.0005549122352940002,0.023096034747190672,0.04619756315527515,                                0.2437077420144654,0.009750302717746615
                
        hits, z_shift, r_inv = params

        
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
        
        
        
        #hits = preprocess_hits(hits, z_shift)
        #hits['z'] =  hits['z'].values + z_shift
        #hits['r'] = np.sqrt(hits['x'].values**2+hits['y'].values**2)        


        #h['r'] = np.sqrt(h['x'].values**2+h['y'].values**2+h['z'].values**2)
#         hits['rt'] = np.sqrt(hits['x'].values**2+hits['y'].values**2)
#         hits['zr'] =  hits['z'].values / hits['rt'].values

#         hits['a0'] = np.arctan2(hits['y'].values,hits['x'].values)     
        hits['tmp'] =  hits['rt'] * r_inv / 2 
        #hits['tmp'] = hits['tmp'].apply(lambda x: min(1, np.abs(x)) * np.sign(x)   )
        
        #hits['theta'] = hits['a0'] + np.arctan( hits['tmp']  )
        #hits['theta'] = hits['a0'] - np.arccos( hits['tmp']  )
        hits['theta'] = hits['a0'] - np.nan_to_num(np.arccos(hits['tmp'].values))
        
        hits['sin_theta'] = np.sin(hits['theta'].values)
        hits['cos_theta'] = np.cos(hits['theta'].values)            
        ss = StandardScaler()

        hits = ss.fit_transform(hits[['sin_theta', 'cos_theta', 'z1', 'z2','x1','x2','x3','x4']].values)
        cx = np.array([w1, w1, w2, w3, w4, w5, w6, w7])
        hits = np.multiply(hits, cx)
        

#         hits['sina1'] = np.sin(hits['a1'].values)
#         hits['cosa1'] = np.cos(hits['a1'].values)

#         ss = StandardScaler()
#         hits = ss.fit_transform(hits[['sin_theta','cos_theta', 'zr']].values) 
        #dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
        #hits = np.multiply(hits, c)
        new_cluster=DBSCAN(eps=0.0048,min_samples=1,metric='euclidean',n_jobs=1).fit(hits).labels_
        return new_cluster

                    
    
        
        
    def Hough_clustering(self,hits, epsilon, n_iter): # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        #merged_cluster = self.cluster

        adaptive_eps_coefficient = 1
        
        result = []
        
        params = []
        for i in range(0, n_iter):

            r_inv = np.random.normal(0.0, 0.001)

            #dz = 0
            z_shift = np.random.normal(0.0, 4.5)
            #z_shift = 0
            #eps_new = epsilon + i*1*10**(-5)

            params.append((hits, z_shift, r_inv))
            
            
        
             
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
#     h['zdivrt'] = h['z'].values/h['rt'].values
#     h['zdivr'] = h['z'].values/h['r'].values
#     h['xdivr'] = h['x'].values / h['r'].values
#     h['ydivr'] = h['y'].values / h['r'].values
    return h



# In[103]:


min_samples_in_cluster = 1

model = Clusterer()
#model.initialize(hits)

#hits_with_dz = preprocess_hits(hits, 0.055*i)

result = model.Hough_clustering(hits,epsilon=0.0048, n_iter=60000)


# In[106]:



# In[107]:


#result = result_53
labels = range(result.shape[1])

#for k in [18,16,14, 12, 10, 8, 6, 4, 0]:
#for k in list(reversed(range(21))):
for k in [0]:
    for i in range(len(result)):
        labels = merge(labels, result[i], k)
  
    submission = create_one_event_submission(0, hits['hit_id'].values, labels)
    print(score_event(truth, submission))
    #submission = extend(submission,hits)
    #submission = extend(submission,hits)

    #print(score_event(truth, submission))
    
    labels = submission['track_id'].values


