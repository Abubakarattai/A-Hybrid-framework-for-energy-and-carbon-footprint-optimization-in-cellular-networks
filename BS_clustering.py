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
from tqdm import tqdm
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
    
print('power of macro based on traffic', power_macro_traffic(1)) 
print('power of RRH based on traffic', power_RRH_traffic(1))  
print('power of micro based on traffic', power_micro_traffic(1)) 
print('power of pico based on traffic', power_pico_traffic(1)) 
print('power of femto based on traffic', power_femto_traffic(1)) 
print('')
    
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
n_sc = 12;
N_c = [0.75, 0.75, 0.75, 0.50, 0.5, 0.5, 0.25, 0.25, 0.25, 0.15, 0.15, 0.15]
l_max = 1

index_cal = [[1],[2],[3],[4]]
if n_sc == 12:
    index_cal = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
else:
    index_cal = [[1],[2],[3],[4]]

#Tf = pd.read_csv('C:\\Users\\2309848A\\Dropbox\\attai\\project_drone\\clust_data.csv', header= None)
Tf = pd.read_csv('clust_data.csv', header= None)
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
for row in tqdm(range(100)):
    data = (np.array(traffic_sc2.iloc[row,:]).reshape(-1,1))
    cluster, no_cluster, cluster_keys, cluster_values = cluster_module(data)
        
    area_cluster = np.reshape(np.array(cluster), (N_BS_x, N_BS_y)) 
    area_cluster_array = np.reshape(np.array(cluster), (N_BS_x*N_BS_y,1)) #
    
    cluster_and_traffic = np.hstack((area_cluster_array,data))
    cluster_and_traffic_df = pd.DataFrame(cluster_and_traffic, columns = ['cluster','traffic'])
    
    
    ab = []
    for j,i in enumerate(cluster_keys):
        ab.append([i, cluster_and_traffic_df.loc[cluster_and_traffic_df['cluster'] == i, 'traffic'].sum() + traffic_mc.loc[row]])
    ab = (np.array(ab).reshape((len(cluster_keys),2)))
    #print(int(np.where(ab[:,1] == ab[:,1][ab[:,1] <=1].max())[0]))
    try :
        remove_cluster = int(np.where(ab[:,1] == ab[:,1][ab[:,1] <=1.0].max())[0])
    except:
        print('not excute')
    #rec_cluster(cluster_and_traffic_df, cluster_keys, cluster_values, -3)
    
    #print('remove cluster', remove_cluster)
    #print('custer to remove is     |' +str(int(ab[remove_cluster,0])))
    P_ma = power_macro_traffic((ab[remove_cluster,1]))
    #print(P_ma)
    print('*****************ab ***********',ab[remove_cluster,0])
    cluster_keys.remove(ab[remove_cluster,0])
    
    
    for i in cluster_keys:
        index = cluster_and_traffic_df.index
        #print(index)
        #print(cluster_and_traffic_df.loc[cluster_and_traffic_df['cluster'] == i, 'traffic'].sum())
        condition = cluster_and_traffic_df['cluster']==i
        k = (index[condition])
        P_sc_T = 0
        for p in k:
            #print(p)
            if 0<= p & p <= 2:
                power_RRH_traffic(traffic_sc.iloc[row,p])
                P_sc_T = P_sc_T  + power_RRH_traffic(traffic_sc.iloc[row,p])
                
            if 3<= p & p <= 5:
                power_micro_traffic(traffic_sc.iloc[row,p])
                P_sc_T  = P_sc_T + power_micro_traffic(traffic_sc.iloc[row,p])
                
            if 6<= p & p <= 8:
                power_pico_traffic(traffic_sc.iloc[row,p])
                P_sc_T = P_sc_T  + power_pico_traffic(traffic_sc.iloc[row,p])
                
            if 9<= p & p <= 11:
                power_femto_traffic(traffic_sc.iloc[row,p])
                P_sc_T  = P_sc_T  + power_femto_traffic(traffic_sc.iloc[row,p])
    #print(P_sc_T)
    
    condition2 = cluster_and_traffic_df['cluster']==(int(ab[remove_cluster,0]))
    r = (index[condition2])
    for u in r:
        #print(u)
        if 0<= u & u <= 2:
            p_rhs = 56
            P_sc_T = P_sc_T  + p_rhs
                
        if 3<= u & u <= 5:
            p_mis = 39
            P_sc_T = P_sc_T  + p_mis
                
        if 6<= u & u <= 8:
            p_pis = 4.3
            P_sc_T  = P_sc_T  + p_pis
                
        if 9<= u & u <= 11:
            p_fes = 2.9
            P_sc_T  = P_sc_T  + p_fes
    #print(P_sc_T)
    P_total= P_ma + P_sc_T
    #print(P_total)
    
    power_saved.append((power_all_on(row, index_cal) -  P_total))
    #print(power_all_on(row, index_cal))
    #print(power_saved)
    
    
                
            
                
    
             
        
        
    
   