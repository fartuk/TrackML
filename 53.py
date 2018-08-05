
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

cnt = 0
for event_id, hits, cells, particles, truth in load_dataset('../storage/track_ml_data/train_5.zip'):
    if cnt == 1:
        break
    cnt += 1






def merge(cl1, cl2): # merge cluster 2 to cluster 1
    d = pd.DataFrame(data={'s1':cl1,'s2':cl2})
    d['N1'] = d.groupby('s1')['s1'].transform('count')
    d['N2'] = d.groupby('s2')['s2'].transform('count')
    maxs1 = d['s1'].max()
    cond = np.where((d['N2'].values>d['N1'].values) & (d['N2'].values<25)) # tìm vị trí hit với nhit của cluster mới > nhits cluster cũ
    s1 = d['s1'].values 
    s1[cond] = d['s2'].values[cond]+maxs1 # gán tất cả các hit đó thuộc về track mới (+maxs1 để tăng label cho track để nó khác ban đầu)
    return s1

def extract_good_hits(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    return tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]

def fast_score(good_hits_df):
    return good_hits_df.weight.sum()


def analyze_truth_perspective(truth, submission):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    good_hits = tru[(tru.count_both > 0.5*tru.count_particle) & (tru.count_both > 0.5*tru.count_track)]
    score = good_hits.weight.sum()
    
    anatru = tru.particle_id.value_counts().value_counts().sort_index().to_frame().rename({'particle_id':'true_particle_counts'}, axis=1)
    #anatru['true_particle_ratio'] = anatru['true_particle_counts'].values*100/np.sum(anatru['true_particle_counts'])

    anatru['good_tracks_counts'] = np.zeros(len(anatru)).astype(int)
    anatru['good_tracks_intersect_nhits_avg'] = np.zeros(len(anatru))
    anatru['best_detect_intersect_nhits_avg'] = np.zeros(len(anatru))
    for nhit in tqdm(range(4,20)):
        particle_list  = tru[(tru.count_particle==nhit)].particle_id.unique()
        intersect_count = 0
        good_tracks_count = 0
        good_tracks_intersect = 0
        for p in particle_list:
            nhit_intersect = tru[tru.particle_id==p].count_both.max()
            intersect_count += nhit_intersect
            corresponding_track = tru.loc[tru[tru.particle_id==p].count_both.idxmax()].track_id
            leng_corresponding_track = len(tru[tru.track_id == corresponding_track])
            
            if (nhit_intersect >= nhit/2) and (nhit_intersect >= leng_corresponding_track/2):
                good_tracks_count += 1
                good_tracks_intersect += nhit_intersect
        intersect_count = intersect_count/len(particle_list)
        anatru.at[nhit,'best_detect_intersect_nhits_avg'] = intersect_count
        anatru.at[nhit,'good_tracks_counts'] = good_tracks_count
        if good_tracks_count > 0:
            anatru.at[nhit,'good_tracks_intersect_nhits_avg'] = good_tracks_intersect/good_tracks_count
    
    return score, anatru, good_hits

def precision(truth, submission,min_hits):
    tru = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    tru['count_both'] = tru.groupby(['track_id', 'particle_id']).hit_id.transform('count')    
    tru['count_particle'] = tru.groupby(['particle_id']).hit_id.transform('count')
    tru['count_track'] = tru.groupby(['track_id']).hit_id.transform('count')
    #print('Analyzing predictions...')
    predicted_list  = tru[(tru.count_track>=min_hits)].track_id.unique()
    good_tracks_count = 0
    ghost_tracks_count = 0
    fp_weights = 0
    tp_weights = 0
    for t in predicted_list:
        nhit_track = tru[tru.track_id==t].count_track.iloc[0]
        nhit_intersect = tru[tru.track_id==t].count_both.max()
        corresponding_particle = tru.loc[tru[tru.track_id==t].count_both.idxmax()].particle_id
        leng_corresponding_particle = len(tru[tru.particle_id == corresponding_particle])
        if (nhit_intersect >= nhit_track/2) and (nhit_intersect >= leng_corresponding_particle/2): #if the predicted track is good
            good_tracks_count += 1
            tp_weights += tru[(tru.track_id==t)&(tru.particle_id==corresponding_particle)].weight.sum()
            fp_weights += tru[(tru.track_id==t)&(tru.particle_id!=corresponding_particle)].weight.sum()
        else: # if the predicted track is bad
                ghost_tracks_count += 1
                fp_weights += tru[(tru.track_id==t)].weight.sum()
    all_weights = tru[(tru.count_track>=min_hits)].weight.sum()
    precision = tp_weights/all_weights*100
    print('Precision: ',precision,', good tracks:', good_tracks_count,', total tracks:',len(predicted_list),
           ', loss:', fp_weights, ', reco:', tp_weights, 'reco/loss', tp_weights/fp_weights)
    return precision


class Clusterer(object):
    def __init__(self):                        
        self.abc = []
          
    def initialize(self,dfhits):
        self.cluster = range(len(dfhits))
        
    def Hough_clustering(self,dfh,coef,epsilon,min_samples=1,n_loop=180,verbose=True): # [phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
        merged_cluster = self.cluster
        mm = 1
        stepii = 0.000005
        count_ii = 0
        adaptive_eps_coefficient = 1
        
        result = []
        
        for ii in np.arange(0, n_loop*stepii, stepii):
            count_ii += 1
            for jj in range(2):
                mm = mm*(-1)
                eps_new = epsilon + count_ii*adaptive_eps_coefficient*10**(-5)
                dfh['a1'] = dfh['a0'].values - np.nan_to_num(np.arccos(mm*ii*dfh['rt'].values))
                dfh['sina1'] = np.sin(dfh['a1'].values)
                dfh['cosa1'] = np.cos(dfh['a1'].values)
                ss = StandardScaler()
                dfs = ss.fit_transform(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']].values) 
                #dfs = scale_ignore_nan(dfh[['sina1','cosa1','zdivrt','zdivr','xdivr','ydivr']])
                dfs = np.multiply(dfs, coef)
                new_cluster=DBSCAN(eps=eps_new,min_samples=min_samples,metric='euclidean',n_jobs=4).fit(dfs).labels_
                merged_cluster = merge(merged_cluster, new_cluster)
                result += [new_cluster]
                if verbose == True:
                    sub = create_one_event_submission(0, hits, merged_cluster)
                    good_hits = extract_good_hits(truth, sub)
                    score_1 = fast_score(good_hits)
                    print('2r0_inverse:', ii*mm ,'. Score:', score_1)
                    #clear_output(wait=True)
        self.cluster = merged_cluster
        return result

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
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





def foo(i):
    model = Clusterer()
    model.initialize(hits)
    
    hits_with_dz = preprocess_hits(hits, 0.055*i)

    result = model.Hough_clustering(hits_with_dz,coef=c,epsilon=0.0048,min_samples=min_samples_in_cluster,
                       n_loop=300,verbose=True)
    
    
    
    second = []
    for k in range(10):

        np.random.shuffle(result)

        #result = res0
        labels = range(result.shape[1])

        for k in [0]:
            for i in range(len(result[:])):
                labels = merge(labels, result[i], k)

            submission = create_one_event_submission(0, hits['hit_id'].values, labels)
            print(score_event(truth, submission))

        second += [labels]
        
    result = np.array(second)
    labels = range(result.shape[1])

    for k in [0]:
        for i in range(len(result[:])):
            labels = merge(labels, result[i], k)

        submission = create_one_event_submission(0, hits['hit_id'].values, labels)
        print(score_event(truth, submission))
    
    
    np.save('predicts/53/{}'.format(i), labels)
    return None




from multiprocessing import Pool

c = [1.5,1.5,0.73,0.17,0.027,0.027] #[phi_coef,phi_coef,zdivrt_coef,zdivr_coef,xdivr_coef,ydivr_coef]
min_samples_in_cluster = 1



pool = Pool(processes=8)

import threading
lock = threading.Lock()


for df in tqdm(pool.imap(foo,  range(-100,101))):
    None
    
pool.close()
















