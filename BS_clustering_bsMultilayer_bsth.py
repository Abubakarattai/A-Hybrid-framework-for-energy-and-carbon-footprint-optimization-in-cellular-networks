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
import time
import pathlib
import os
#
def cluster_locator(X, n_max):
    
    if n_max > 11:
        N_max = 11
    else:
        N_max = n_max - 1
    #print('Nmax is ', N_max)    
    #if n_max < 4:
    #    return 2
    #N_max = 11
    #N_max = n_max - 1

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "max_iter": 500,
        "random_state": 42,
    }
    
    sse = []
    for k in range(1, N_max):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
        
    kl = KneeLocator(range(1, N_max), sse, curve="convex", direction="decreasing")
    
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

    return P_all_on-(p_bs_off+p_bs_on), p_bs_off
    

def cluster_module(data, n_max):
    if n_max > 7:
        no_cluster = cluster_locator(data, n_max)
        kmeans  = KMeans(n_clusters = no_cluster, init='k-means++', random_state= random_state)
        cluster = kmeans.fit_predict(data)
        cluster_keys = list(Counter(list(cluster)).keys()) # equals to list(set(words))
        cluster_values = list(Counter(list(cluster)).values()) # counts the elements' frequency
    else:
        no_cluster = 2
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

    as_list = combinate_info.index.tolist()
    idx = as_list.index(())
    as_list[idx] = (-1,)
    combinate_info.index = as_list
    
    
    combinate_info['combination'] = combinate_info.index
    combinate_info = combinate_info[(combinate_info.traffic<=1) & (combinate_info.traffic>0)]
    combinate_info['indexCount'] = [len(c) for c in combinate_info['combination']] 
    out = combinate_info.loc[(combinate_info['traffic']<=1)]

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
def multi_cluster(cluster_and_traffic_df, row, x):

    #print('unique', cluster_and_traffic_df.cluster.unique())
    #print('count the occorance', cluster_and_traffic_df['cluster'].value_counts())
    cluster_and_traffic_df['freq'] = cluster_and_traffic_df.groupby('cluster')['cluster'].transform('count')
    #print('after cluster', cluster_and_traffic_df)
    #print('before cluster', cluster_and_traffic_df)
    cluster_df = cluster_and_traffic_df[cluster_and_traffic_df["freq"] <= n_bs_thr]
    #print(cluster_df)
    
    if len(cluster_df) > 0:
        
        cluster_keys = cluster_df.cluster.unique()
        for kQ in cluster_keys:
            results_df = cluster_df[cluster_df['cluster'] == kQ]
            cluster_remove, total_traffic = enumerate_bs(results_df, row)
            for index_i, i in enumerate(cluster_remove):
                #print(storage_energy)
                storage_energy.append(power_module(i, row, total_traffic[index_i])[0])
            #print('****** anapther cluster **********')
            #print(cluster_remove)
        cluster_and_traffic_df = cluster_and_traffic_df[cluster_and_traffic_df["freq"] > n_bs_thr]

    cluster_and_traffic_df = cluster_and_traffic_df.drop('freq', 1)


    if (len(cluster_and_traffic_df) == 0) :

        
        return max(storage_energy)
    
    else:
        #print(cluster_and_traffic_df)
        max_index_cluster = 0
        new_cluster = []
        c_and_t_df = cluster_and_traffic_df.copy()
        for i in cluster_and_traffic_df.cluster.unique():
            subset_df = cluster_and_traffic_df[cluster_and_traffic_df["cluster"] == i]
            data = (np.array(subset_df.loc[:,'traffic']).reshape(-1,1))
 

            cluster, no_cluster, cluster_keys, cluster_values = cluster_module(data,data.shape[0])
               

            cluster = [x+max_index_cluster for x in cluster]

            
            for ii,jj in enumerate(list(subset_df.index)):
                c_and_t_df.loc[jj,'cluster']= cluster[ii]

            
            max_index_cluster = max(cluster) + 1
 
        cluster_and_traffic_df = c_and_t_df
        x = x-1

        
        
        return multi_cluster(cluster_and_traffic_df, row,  x)  

    
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
#n_sc = 24;   # change number of small cell

for n_sc in [60]:
#for n_sc in [96]:

    N_c = generate_N_c(n_sc)
    index_cal = (gen_index_call(n_sc))
    l_max = 1
    n_bs_thr = 12
    
    # def multi_bt(n_sc):    
    #     file = pathlib.Path('data/clust_data_'+str(n_sc))  
        
    #     for path in os.listdir(file):
    #         full_path = os.path.join(file, path)
    #         print('full path', full_path)
    #         if os.path.isfile(full_path):
    #             print (full_path)
        
    #     return file 
    
    
    
    file_name_sc = pathlib.Path('data/clust_data_'+str(n_sc))  
        
    for path_sc in os.listdir(file_name_sc):
        full_path = os.path.join(file_name_sc, path_sc)
        print(full_path)
        Tf = pd.read_csv(full_path, header= None)
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
        
        file = pathlib.Path('results/databsth_'+str(n_sc)+'.csv' )  
        if file.exists ():
            df_n_sc= pd.read_csv('results/databsth_'+str(n_sc)+'.csv' )
        else:
            columns = ['time_taken','sum_power_saved']
            df_n_sc = pd.DataFrame(columns=columns)
        
        start = time.time()
        # for row in tqdm(range(rows)):
        for row in (range(rows)):
        #for row in [102]:
            #print(traffic_mc.loc[row])
            storage_index = []
            traffic_index = []
            data = (np.array(traffic_sc2.iloc[row,:]).reshape(-1,1))
        
            
            cluster, no_cluster, cluster_keys, cluster_values = cluster_module(data, data.shape[0])
            #print('number of cluster', no_cluster)
                
            area_cluster_array = np.reshape(np.array(cluster), (len(cluster),1)) #
            cluster_and_traffic = np.hstack((area_cluster_array,data))
            cluster_and_traffic_df = pd.DataFrame(cluster_and_traffic, columns = ['cluster','traffic'])
            
            storage_energy = []
        
        
                
            #print(multi_cluster(cluster_and_traffic_df, row,  2))
            storage_energy_total.append(multi_cluster(cluster_and_traffic_df, row,  2))
          
        
        end = time.time()
        new_row = {'time_taken':(end - start), 'sum_power_saved':sum(storage_energy_total)}
        df_n_sc = df_n_sc.append(new_row, ignore_index=True)
        
        df_n_sc.to_csv('results/databsth_'+str(n_sc)+'.csv', index=False) 
        
        #%%    
        
        nn = rows  
        
        bs_number = [8, 12, 16, 20, 24, 36, 48, 60, 72, 84, 96, 108, 120]
        columns = ['time_taken','sum_power_saved']
        dict={'bs_number':bs_number,'time_taken':[0 for i in range(len(bs_number))],'sum_power_saved':[0 for i in range(len(bs_number))]}
        out = pd.DataFrame(dict, columns = ['bs_number', 'time_taken','sum_power_saved'])
        out = out.set_index('bs_number')
        #print(out.head())
        
        
        
        
        #df.set_index('month')
        
        for i in bs_number:
            file = pathlib.Path('results/databsth_'+str(i)+'.csv')
            if file.exists ():
                #print(i)
                df_n_sc= pd.read_csv('results/databsth_'+str(i)+'.csv')
                out.at[i, 'time_taken'] = df_n_sc['time_taken'].mean()
                out.at[i, 'sum_power_saved'] = df_n_sc['sum_power_saved'].mean()
                #out.iloc[i,0] = df_n_sc['time_taken'].mean()
                #out.iloc[i,1] = df_n_sc['sum_power_saved'].mean()
        
        #print(out)
        #out.to_csv('out/output_all.csv', index=False) 
        out.to_csv('out/output_alldatabsth.csv') 
        #print(full_path)
        print('out see', sum(storage_energy_total))
        
            
        # #df = pd.read_csv('Pwr_12Scs_ES_clust.csv', header= None)    
        # df = pd.read_csv('Pwr_'+str(n_sc)+'Scs_ES_clust.csv', header= None)  
        # #x = [i for i in range(len(df))]  
        # x = [i for i in range(nn)] 
        # #print(df.shape[1])
        # y = df.iloc[0:nn,df.shape[1]-1]    
        
        # plt.plot(x,y)      
        # plt.plot(x,storage_energy_total) 
    
    #print(multi_bt(12))
            
                
    
             
        
        
    
   