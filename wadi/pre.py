# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder





df0 = pd.read_csv("WADI_14days.csv",header=0)

df1 = pd.read_csv("WADI_attackdata.csv",header=0)


str1='\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\'


for i in ['Row','Date','Time',str1+'LEAK_DIFF_PRESSURE',str1+'PLANT_START_STOP_LOG',str1+'TOTAL_CONS_REQUIRED_FLOW']:
    df0.pop(i)
    df1.pop(i)

df0['Label']=0
df1['Label']=1

key_df0 = df0.keys().tolist()
key_df1 = df1.keys().tolist()


for i, v in enumerate(key_df0):
    key_df0[i]=key_df0[i].replace(str1,'')
df0.columns = key_df0

for i, v in enumerate(key_df1):
    key_df1[i]=key_df1[i].replace(str1,'')
df1.columns = key_df1

df = pd.concat( [df0.sample(n=500000, random_state=1), df1], axis=0 )
#df = pd.concat( [df0, df1], axis=0 )

df=df.sample(n=100000, random_state=1)



print('load ok')

key_df = df.keys().tolist()

matrix_key = key_df
matrix_key.pop(-1)


#matrix = [[0 for i in range(len(matrix_key))] for j in range(len(matrix_key))]
#def matrix_link2(m,k,a,b):
#    m[k.index(a)][k.index(b)]=1
#    m[k.index(b)][k.index(a)]=1



import networkx as nx
G=nx.Graph()
for i in matrix_key:
    G.add_node(i)
    
def matrix_link(m,k,a,b):
    m.add_edge(a,b)

matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_201_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_301_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_401_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_301_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_401_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_301_PV','2_FQ_401_PV')
matrix_link(G,matrix_key,'2_FQ_301_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_301_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_401_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_401_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_501_PV','2_FQ_601_PV')


matrix_link(G,matrix_key,'2_SV_101_STATUS','2_SV_201_STATUS')
matrix_link(G,matrix_key,'2_SV_101_STATUS','2_SV_301_STATUS')
matrix_link(G,matrix_key,'2_SV_101_STATUS','2_SV_401_STATUS')
matrix_link(G,matrix_key,'2_SV_101_STATUS','2_SV_501_STATUS')
matrix_link(G,matrix_key,'2_SV_101_STATUS','2_SV_601_STATUS')
matrix_link(G,matrix_key,'2_SV_201_STATUS','2_SV_301_STATUS')
matrix_link(G,matrix_key,'2_SV_201_STATUS','2_SV_401_STATUS')
matrix_link(G,matrix_key,'2_SV_201_STATUS','2_SV_501_STATUS')
matrix_link(G,matrix_key,'2_SV_201_STATUS','2_SV_601_STATUS')
matrix_link(G,matrix_key,'2_SV_301_STATUS','2_SV_401_STATUS')
matrix_link(G,matrix_key,'2_SV_301_STATUS','2_SV_501_STATUS')
matrix_link(G,matrix_key,'2_SV_301_STATUS','2_SV_601_STATUS')
matrix_link(G,matrix_key,'2_SV_401_STATUS','2_SV_501_STATUS')
matrix_link(G,matrix_key,'2_SV_401_STATUS','2_SV_601_STATUS')
matrix_link(G,matrix_key,'2_SV_501_STATUS','2_SV_601_STATUS')


matrix_link(G,matrix_key,'2_MCV_101_CO','2_MCV_201_CO')
matrix_link(G,matrix_key,'2_MCV_101_CO','2_MCV_301_CO')
matrix_link(G,matrix_key,'2_MCV_101_CO','2_MCV_401_CO')
matrix_link(G,matrix_key,'2_MCV_101_CO','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MCV_101_CO','2_MCV_601_CO')
matrix_link(G,matrix_key,'2_MCV_201_CO','2_MCV_301_CO')
matrix_link(G,matrix_key,'2_MCV_201_CO','2_MCV_401_CO')
matrix_link(G,matrix_key,'2_MCV_201_CO','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MCV_201_CO','2_MCV_601_CO')
matrix_link(G,matrix_key,'2_MCV_301_CO','2_MCV_401_CO')
matrix_link(G,matrix_key,'2_MCV_301_CO','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MCV_301_CO','2_MCV_601_CO')
matrix_link(G,matrix_key,'2_MCV_401_CO','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MCV_401_CO','2_MCV_601_CO')
matrix_link(G,matrix_key,'2_MCV_501_CO','2_MCV_601_CO')

matrix_link(G,matrix_key,'2_MV_101_STATUS','2_MV_201_STATUS')
matrix_link(G,matrix_key,'2_MV_101_STATUS','2_MV_301_STATUS')
matrix_link(G,matrix_key,'2_MV_101_STATUS','2_MV_401_STATUS')
matrix_link(G,matrix_key,'2_MV_101_STATUS','2_MV_501_STATUS')
matrix_link(G,matrix_key,'2_MV_101_STATUS','2_MV_601_STATUS')
matrix_link(G,matrix_key,'2_MV_201_STATUS','2_MV_301_STATUS')
matrix_link(G,matrix_key,'2_MV_201_STATUS','2_MV_401_STATUS')
matrix_link(G,matrix_key,'2_MV_201_STATUS','2_MV_501_STATUS')
matrix_link(G,matrix_key,'2_MV_201_STATUS','2_MV_601_STATUS')
matrix_link(G,matrix_key,'2_MV_301_STATUS','2_MV_401_STATUS')
matrix_link(G,matrix_key,'2_MV_301_STATUS','2_MV_501_STATUS')
matrix_link(G,matrix_key,'2_MV_301_STATUS','2_MV_601_STATUS')
matrix_link(G,matrix_key,'2_MV_401_STATUS','2_MV_501_STATUS')
matrix_link(G,matrix_key,'2_MV_401_STATUS','2_MV_601_STATUS')
matrix_link(G,matrix_key,'2_MV_501_STATUS','2_MV_601_STATUS')


matrix_link(G,matrix_key,'2_MV_101_STATUS','2_MCV_101_CO')
matrix_link(G,matrix_key,'2_MV_201_STATUS','2_MCV_201_CO')
matrix_link(G,matrix_key,'2_MV_301_STATUS','2_MCV_301_CO')
matrix_link(G,matrix_key,'2_MV_401_STATUS','2_MCV_401_CO')
matrix_link(G,matrix_key,'2_MV_501_STATUS','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MV_601_STATUS','2_MCV_601_CO')

matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MCV_101_CO')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MCV_201_CO')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MCV_301_CO')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MCV_401_CO')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MCV_601_CO')

matrix_link(G,matrix_key,'2_MV_005_STATUS','2_MCV_101_CO')
matrix_link(G,matrix_key,'2_MV_005_STATUS','2_MCV_201_CO')
matrix_link(G,matrix_key,'2_MV_005_STATUS','2_MCV_301_CO')
matrix_link(G,matrix_key,'2_MV_005_STATUS','2_MCV_401_CO')
matrix_link(G,matrix_key,'2_MV_005_STATUS','2_MCV_501_CO')
matrix_link(G,matrix_key,'2_MV_005_STATUS','2_MCV_601_CO')


matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MV_005_STATUS')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MV_002_STATUS')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MV_004_STATUS')

matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MV_006_STATUS')
matrix_link(G,matrix_key,'2_MV_006_STATUS','2_MV_006_STATUS')

matrix_link(G,matrix_key,'2_MV_001_STATUS','2_MV_002_STATUS')
matrix_link(G,matrix_key,'2_MV_003_STATUS','2_MV_004_STATUS')
matrix_link(G,matrix_key,'2_MV_001_STATUS','2_MV_003_STATUS')
matrix_link(G,matrix_key,'2_MV_002_STATUS','2_MV_004_STATUS')

matrix_link(G,matrix_key,'1_MV_001_STATUS','3_MV_002_STATUS')


matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_201_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_301_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_401_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_101_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_301_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_401_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_201_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_301_PV','2_FQ_401_PV')
matrix_link(G,matrix_key,'2_FQ_301_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_301_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_401_PV','2_FQ_501_PV')
matrix_link(G,matrix_key,'2_FQ_401_PV','2_FQ_601_PV')
matrix_link(G,matrix_key,'2_FQ_501_PV','2_FQ_601_PV')



matrix_link(G,matrix_key,'2_LT_001_PV','2_LT_002_PV')


matrix_link(G,matrix_key,'3_LS_001_AL','2_LS_101_AL')
matrix_link(G,matrix_key,'3_LS_001_AL','2_LS_201_AL')
matrix_link(G,matrix_key,'3_LS_001_AL','2_LS_301_AL')
matrix_link(G,matrix_key,'3_LS_001_AL','2_LS_401_AL')
matrix_link(G,matrix_key,'3_LS_001_AL','2_LS_501_AL')
matrix_link(G,matrix_key,'3_LS_001_AL','2_LS_601_AL')

matrix_link(G,matrix_key,'2_LS_101_AH','2_LS_201_AH')
matrix_link(G,matrix_key,'2_LS_101_AH','2_LS_301_AH')
matrix_link(G,matrix_key,'2_LS_101_AH','2_LS_401_AH')
matrix_link(G,matrix_key,'2_LS_101_AH','2_LS_501_AH')
matrix_link(G,matrix_key,'2_LS_101_AH','2_LS_601_AH')
matrix_link(G,matrix_key,'2_LS_201_AH','2_LS_301_AH')
matrix_link(G,matrix_key,'2_LS_201_AH','2_LS_401_AH')
matrix_link(G,matrix_key,'2_LS_201_AH','2_LS_501_AH')
matrix_link(G,matrix_key,'2_LS_201_AH','2_LS_601_AH')
matrix_link(G,matrix_key,'2_LS_301_AH','2_LS_401_AH')
matrix_link(G,matrix_key,'2_LS_301_AH','2_LS_501_AH')
matrix_link(G,matrix_key,'2_LS_301_AH','2_LS_601_AH')
matrix_link(G,matrix_key,'2_LS_401_AH','2_LS_501_AH')
matrix_link(G,matrix_key,'2_LS_401_AH','2_LS_601_AH')
matrix_link(G,matrix_key,'2_LS_501_AH','2_LS_601_AH')

matrix_link(G,matrix_key,'2_LS_101_AL','2_LS_201_AL')
matrix_link(G,matrix_key,'2_LS_101_AL','2_LS_301_AL')
matrix_link(G,matrix_key,'2_LS_101_AL','2_LS_401_AL')
matrix_link(G,matrix_key,'2_LS_101_AL','2_LS_501_AL')
matrix_link(G,matrix_key,'2_LS_101_AL','2_LS_601_AL')
matrix_link(G,matrix_key,'2_LS_201_AL','2_LS_301_AL')
matrix_link(G,matrix_key,'2_LS_201_AL','2_LS_401_AL')
matrix_link(G,matrix_key,'2_LS_201_AL','2_LS_501_AL')
matrix_link(G,matrix_key,'2_LS_201_AL','2_LS_601_AL')
matrix_link(G,matrix_key,'2_LS_301_AL','2_LS_401_AL')
matrix_link(G,matrix_key,'2_LS_301_AL','2_LS_501_AL')
matrix_link(G,matrix_key,'2_LS_301_AL','2_LS_601_AL')
matrix_link(G,matrix_key,'2_LS_401_AL','2_LS_501_AL')
matrix_link(G,matrix_key,'2_LS_401_AL','2_LS_601_AL')
matrix_link(G,matrix_key,'2_LS_501_AL','2_LS_601_AL')

matrix_link(G,matrix_key,'2_LS_001_AL','2_LS_002_AL')
matrix_link(G,matrix_key,'1_LS_001_AL','1_LS_002_AL')


matrix_link(G,matrix_key,'2A_AIT_001_PV','2B_AIT_001_PV')
matrix_link(G,matrix_key,'2A_AIT_002_PV','2B_AIT_002_PV')
matrix_link(G,matrix_key,'2A_AIT_003_PV','2B_AIT_003_PV')
matrix_link(G,matrix_key,'2A_AIT_004_PV','2B_AIT_004_PV')

import matplotlib.pyplot as plt
nx.draw(G,pos = nx.random_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =18,node_size =20)
plt.show()


dc=nx.degree_centrality(G)
bc=nx.betweenness_centrality(G)
cc=nx.closeness_centrality(G)

import pickle

for j in matrix_key:
    df[[j]] = df[[j]].astype(float)


dc=nx.degree_centrality(G)
bc=nx.betweenness_centrality(G)
cc=nx.closeness_centrality(G)

mc={}
for k in dc:
    mc[k]=(dc[k]+bc[k]+cc[k])

ccc={'dc':dc,'bc':bc,'cc':cc,'mc':mc}



for j in matrix_key:
    df[[j]] = df[[j]].astype(float)





ddd={}


from tqdm import tqdm

for qqq in ccc:
    c2use=ccc[qqq]
    print('start'+'qqq')
    
    t=df.copy()
 
    for index,row in tqdm(t.iterrows(),total=t.shape[0]):
        tmp={}
        for j in matrix_key:
            tmp[j]=row[j]*c2use[j]
        for j in matrix_key:
            tmp_v=row[j]
    #        print(j)
    #        print(tmp_v)
            for jj in G[j]:
                tmp_v += 5*tmp[jj] #xxx
    #            print(tmp[jj])
            
    #        print(tmp_v)
    #        print('=================')
    #        print(row[j])
    #        row[j] = tmp_v
            t.set_value(index,j,tmp_v)
    #        print(row[j])
    
    pickle.dump(t,open(qqq,'wb'))
    print('saved'+'qqq')