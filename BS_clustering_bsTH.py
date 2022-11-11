# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 18:59:50 2021

@author: 2309848A
"""
# Base Station Clustering
# import libraries
import numpy as np 
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator
from collections import Counter
from itertools import chain, combinations
from tqdm import tqdm
import matplotlib.pyplot as plt
#
def cluster_locator(X):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 500,
        "random_state": 42,
    }
    
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
        
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    
    return kl.elbow

def power_all_on(row_index, index_cal):
    
    P_ma_1 = p_mao + k_ma*Tf.iloc[row_index,0]*p_matx
    P_sc_1 = 0
    for i in index_cal[0]:
        P_sc_1 = P_sc_1 + (p_rho + k_rh*Tf.iloc[row_index, i]*p_rhtx);
    for i in index_cal[1]:
        P_sc_1 = P_sc_1 + (p_mio + k_mi*Tf.iloc[row_index, i]*p_mitx);
    for i in index_cal[2]:
        P_sc_1 = P_sc_1 + (p_pio + k_pi*Tf.iloc[row_index, i]*p_pitx);
    for i in index_cal[3]:
        P_sc_1 = P_sc_1 + (p_feo + k_fe*Tf.iloc[row_index, i]*p_fetx);
    
    return P_ma_1+P_sc_1

def power_macro_traffic(l_ma):
    p_mao, k_ma, p_matx = 130, 4.7, 20
    
    return p_mao + k_ma*l_ma*p_matx

def power_RRH_traffic(l_rh):
    p_rho, k_rh, p_rhtx, p_rhs = 84, 2.8, 20, 56
    
    return p_rho + k_rh*l_rh*p_rhtx

def power_micro_traffic(l_mi):
    p_mio, k_mi, p_mitx, P_mis = 56, 2.6, 6.3, 39.0

    return p_mio + k_mi*l_mi*p_mitx

def power_pico_traffic(l_pi):
    p_pio, k_pi, p_pitx, P_pis = 6.8, 4.0, 0.13, 4.3

    return p_pio + k_pi*l_pi*p_pitx

def power_femto_traffic(l_fe):
    p_feo, k_fe, p_fetx, P_fes = 4.8, 8.0, 0.05, 2.9

    return p_feo + k_fe*l_fe*p_fetx

def power_off(u):
    P_sc_T = 0
     
    if (u>= 0) & (u <= int(n_sc/4)-1):
        p_rhs = 56
        P_sc_T = P_sc_T  + p_rhs
                
    if (u>= int(n_sc/4)) & (u <= int(n_sc/2)-1):
        p_mis = 39
        P_sc_T = P_sc_T  + p_mis
                
    if (u>= int(n_sc/2)) & (u <= int(3*n_sc/4)-1):
        p_pis = 4.3
        P_sc_T  = P_sc_T  + p_pis
                
    if (u>= int(3*n_sc/4)) & (u <= n_sc-1):
        p_fes = 2.9
        P_sc_T  = P_sc_T  + p_fes
        
    return  P_sc_T

def power_on(p, row, traffic):
    
    P_sc_T = 0
    if (p>= 0) & (p <= int(n_sc/4)-1):
        
        P_sc_T = P_sc_T  + power_RRH_traffic(traffic_sc.iloc[row,p])
                
    if (p>= int(n_sc/4)) & (p <= int(n_sc/2)-1):

        P_sc_T  = P_sc_T + power_micro_traffic(traffic_sc.iloc[row,p])
        
    if (p>= int(n_sc/2)) & (p <= int(3*n_sc/4)-1):

        P_sc_T = P_sc_T  + power_pico_traffic(traffic_sc.iloc[row,p])
        
    if (p>= int(3*n_sc/4)) & (p <= n_sc-1):

        P_sc_T  = P_sc_T  + power_femto_traffic(traffic_sc.iloc[row,p])
        
    return P_sc_T

def power_module(input_index, row, traffic):
    
    if input_index == -1:
        return 0
    
    all_bs  = set([i for i in range(n_sc)])
    P_all_on = power_all_on(row, index_cal)
    
    p_bs_off = 0
    
    for i in input_index:
        p_bs_off = p_bs_off + power_off(i)
    
    all_lef_on = list(all_bs.difference(set(input_index)))
    
    p_bs_on = p_mao + k_ma*traffic*p_matx
    #print('all_lef_on')
    for i in all_lef_on:
        p_bs_on = p_bs_on+ power_on(i, row, traffic)
        
    # print('**************')
    # print('index', input_index)
    # print('traffic', traffic)
    # print('bs macro switchoff', p_mao + k_ma*traffic*p_matx)
    # print('all off', p_bs_off)
    # print('inde of only on', p_bs_on- (p_mao + k_ma*traffic*p_matx))
    # print('*************')
    #print('power saved   ', P_all_on-(p_bs_off+p_bs_on))
    return P_all_on-(p_bs_off+p_bs_on), p_bs_off
    

def cluster_module(data):
    no_cluster = cluster_locator(data)
    kmeans  = KMeans(n_clusters = no_cluster, init='k-means++', random_state= random_state)
    cluster = kmeans.fit_predict(data)
    cluster_keys = list(Counter(list(cluster)).keys()) # equals to list(set(words))
    cluster_values = list(Counter(list(cluster)).values()) # counts the elements' frequency
    
    return cluster, no_cluster, cluster_keys, cluster_values

def rec_cluster(dataframe, cluster_key, cluster_values, cluster_index):
    
    if (cluster_index)>=1:
        return cluster_index
    else:
        ab=[]
        for j,i in enumerate(cluster_keys):
            ab.append([i, cluster_and_traffic_df.loc[cluster_and_traffic_df['cluster'] == i, 'traffic'].sum() + traffic_mc.loc[row]])
        ab = (np.array(ab).reshape((len(cluster_keys),2)))
        
        try :
            remove_cluster = int(np.where(ab[:,1] == ab[:,1][ab[:,1] <=1].max())[0])
        except:
            results_df = cluster_and_traffic_df[cluster_and_traffic_df['cluster'] == i]
            print('not excute')
        
        return rec_cluster(dataframe, cluster_key, cluster_values, cluster_index)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))  

def enumerate_bs(cluster_df, row):
        
    comb_rows = [l for _, l in enumerate(powerset((cluster_df.index)))]
    combinate_info = pd.DataFrame([cluster_df.loc[c,:].sum() for c in comb_rows], index=comb_rows)
    #print(combinate_info.head())
    combinate_info.loc[:,'traffic'] = combinate_info.loc[:,'traffic'] + traffic_mc.loc[row]
    #combinate_info['traffic'] = combinate_info['traffic'] + traffic_mc.loc[row]
    #print(combinate_info.head())
    as_list = combinate_info.index.tolist()
    idx = as_list.index(())
    as_list[idx] = (-1,)
    combinate_info.index = as_list
    
    
    combinate_info['combination'] = combinate_info.index
    combinate_info = combinate_info[(combinate_info.traffic<=1) & (combinate_info.traffic>0)]
    combinate_info['indexCount'] = [len(c) for c in combinate_info['combination']] 
    out = combinate_info.loc[(combinate_info['traffic']<=1)]
    #as_list = out.index.tolist()
    #idx = as_list.index(())
    #as_list[idx] = (1000)
    #out.index = as_list
    #print(out)
    
    
    #out = combinate_info.loc[(combinate_info['traffic']<=1) & (combinate_info.indexCount >= 1)]
    #out = combinate_info.loc[(combinate_info['traffic']<=1) & (combinate_info.indexCount == combinate_info.indexCount.max()) ]
    #out = out.loc[out.traffic == out.traffic.max()]
    #print(out['combination'])
    #print(out.combination.values)
    #print(out.traffic.values)

    #return out['combination'], out['traffic']  
    return out.combination.values, out.traffic.values
#**************************************************************************
def gen_index_call(n_sc): # the function to generate index of bs
    index_cal = [i+1 for i in range(n_sc)]
    index_cal = np.reshape(index_cal, (4,int(n_sc/4)))
    return index_cal.tolist()
#*****************************************************************************

#*************** generate N_c************************************************ 
def generate_N_c(n_sc):

    n_times = int(n_sc/4)
    
    N_c= [0.75]*n_times + [0.5]*n_times + [0.25]*n_times + [0.15]*n_times

    return N_c
#*****************generate N_c***********************************************
    
# Function for training the UCB
#%power model parameters
p_mao = 130; k_ma = 4.7; p_matx = 20;
p_mio = 56;  k_mi = 2.6; p_mitx = 6.3;  P_mis = 39.0;
p_pio = 6.8; k_pi = 4.0; p_pitx = 0.13; P_pis = 4.3;
p_feo = 4.8; k_fe = 8.0; p_fetx = 0.05; P_fes = 2.9;
p_rho = 84;  k_rh = 2.8; p_rhtx = 20;   P_rhs = 56;  

# initialize number of BS in x and Y
N_BS_x = 3
N_BS_y = 4
random_state = 42
BS_length = 200
n_sc = 20;

N_c = generate_N_c(n_sc)
index_cal = (gen_index_call(n_sc))
l_max = 1
n_bs_thr = 5 


#Tf = pd.read_csv('C:\\Users\\2309848A\\Dropbox\\attai\\project_drone\\clust_data.csv', header= None)
Tf = pd.read_csv('clust_data_'+str(n_sc)+'.csv', header= None)

traffic_mc = Tf.iloc[:,0]
traffic_sc = Tf.iloc[:,1:]
traffic_sc2 = np.multiply(traffic_sc,N_c)

position_bs_x = [(BS_length/2 + BS_length*i) for i in range(N_BS_x)]
position_bs_y = [(BS_length/2 + BS_length*i) for i in range(N_BS_y)]
#position = np.array([position_bs_x, position_bs_y]).reshape((2, len(position_bs_x)))

position = []
for i in position_bs_x:
    for j in position_bs_y:
        position.append([i,j])
        
position = np.array(position).reshape((len(position_bs_x)*len(position_bs_y),2))

power_saved = []
#for row in range(len(traffic_data)):
storage_energy_total = []
rows = 144
for row in tqdm(range(rows)):
#for row in [0]:
    data = (np.array(traffic_sc2.iloc[row,:]).reshape(-1,1))
    #data = (np.array(traffic_sc.iloc[row,:]).reshape(-1,1))
    cluster, no_cluster, cluster_keys, cluster_values = cluster_module(data)
        
    #area_cluster = np.reshape(np.array(cluster), (N_BS_x, N_BS_y)) 
    area_cluster_array = np.reshape(np.array(cluster), (n_sc,1)) #
    
    cluster_and_traffic = np.hstack((area_cluster_array,data))
    cluster_and_traffic_df = pd.DataFrame(cluster_and_traffic, columns = ['cluster','traffic'])
    
    def see_rec(cluster_and_traffic_df, cluster_keys):
        
        if max(cluster_keys)<= n_bs_thr:
            
            return 0
        else:
            return 1
    
    storage_energy = []
    
    for kQ in cluster_keys:

        results_df = cluster_and_traffic_df[cluster_and_traffic_df['cluster'] == kQ]

        cluster_remove, total_traffic = enumerate_bs(results_df, row)

        for index_i, i in enumerate(cluster_remove):

            storage_energy.append(power_module(i, row, total_traffic[index_i])[0])
    
    #print(max(storage_energy))
    storage_energy_total.append(max(storage_energy))
        #for j in i:
        #    print(i)

    
    

nn = rows  

    
#df = pd.read_csv('Pwr_12Scs_ES_clust.csv', header= None)    
df = pd.read_csv('Pwr_'+str(n_sc)+'Scs_ES_clust.csv', header= None)  
#x = [i for i in range(len(df))]  
x = [i for i in range(nn)] 
#print(df.shape[1])
y = df.iloc[0:nn,df.shape[1]-1]    

plt.plot(x,y)      
plt.plot(x,storage_energy_total) 
            
                
    
             
        
        
    
   