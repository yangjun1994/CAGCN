# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


df = pd.read_csv("swatall.csv",header=0)

key_df = df.keys().tolist()


for i, v in enumerate(key_df):
    key_df[i]=key_df[i].replace(' ','')
    if key_df[i] == 'Normal/Attack':
        key_df[i] = 'Label'

df.columns = key_df






print('load ok')



matrix_key = key_df
matrix_key.pop(0)
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

matrix_link(G,matrix_key,'MV101','LIT101')
matrix_link(G,matrix_key,'FIT101','LIT101')
matrix_link(G,matrix_key,'LIT101','P101')
matrix_link(G,matrix_key,'P101','FIT201')
matrix_link(G,matrix_key,'P101','AIT201')
matrix_link(G,matrix_key,'FIT201','P201')
matrix_link(G,matrix_key,'AIT201','P201')
matrix_link(G,matrix_key,'P201','P203')
matrix_link(G,matrix_key,'P203','P205')
matrix_link(G,matrix_key,'P205','MV201')
matrix_link(G,matrix_key,'MV201','AIT202')
matrix_link(G,matrix_key,'MV201','AIT203')
matrix_link(G,matrix_key,'AIT202','LIT301')
matrix_link(G,matrix_key,'AIT203','LIT301')
matrix_link(G,matrix_key,'LIT301','P301')
matrix_link(G,matrix_key,'P301','DPIT301')
matrix_link(G,matrix_key,'DPIT301','LIT401')
matrix_link(G,matrix_key,'LIT401','P401')
matrix_link(G,matrix_key,'P401','FIT401')
matrix_link(G,matrix_key,'FIT401','AIT402')
matrix_link(G,matrix_key,'AIT402','AIT503')
matrix_link(G,matrix_key,'AIT503','P501')
matrix_link(G,matrix_key,'P501','AIT504')
matrix_link(G,matrix_key,'P501','P602')
matrix_link(G,matrix_key,'P602','DPIT301')

import matplotlib.pyplot as plt
import lightgbm as clflib
nx.draw(G,pos = nx.random_layout(G),node_color = 'b',edge_color = 'r',with_labels = True,font_size =18,node_size =20)
plt.show()


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



for qqq in ccc:
    c2use=ccc[qqq]
    
    
    t=df
    
    
    
    
    
    
    
    
    for index,row in t.iterrows():
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
    
    
    print('c changes')
    
    
    #df=t
    
    
    
    
    
    
    
    
    
    
    
    params = {
                'learning_rate': 0.01,
                #'boosting_type': 'dart',
    #            'objective': 'multiclass',
                #'num_class':3,
    #            'metric': 'multi_logloss',
                'num_class':2,
                'metric': 'binary_logloss',
                #'colsample_bytree': 0.8,
                #'max_depth': 30,
                'min_child_weight':1,
                #'n_estimators': 80,
                #'num_leaves': 128,
                #'scale_pos_weight':s,
                #'min_hessian': 1,
                #'verbose': -1,
                #'colsample_bytree': 0.8,
                #'max_depth': 3,
                #'min_child_weight': 0.5,
                #'n_estimators': 100,
                #'num_leaves': 64,
                #'subsample' : 0.8,
                #'colsample_bytree':0.8,
                #'metric': 'auc',
                #'sub_feature': 0.7,
                #'num_leaves': 60,
                #'colsample_bytree': 0.7,
                #'feature_fraction': 0.7,
               # 'min_data': 10,
               # 'min_hessian': 1,
                'verbose': -1,
                }
    
    
    
    
    cf=['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','LIT301','AIT401','AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','PIT501','PIT502','PIT503','FIT601']
    
    #one_hot_feature=['MV101','P101','P102','MV201','P201','P202','P203','P204','P205','P206','MV301','MV302','MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603', 'hour']
    
    one_hot_feature=['MV101','P101','P102','MV201','P201','P202','P203','P204','P205','P206','MV301','MV302','MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603'] #del hour
    
    
    
    data=t
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    #data['hour'] = data['Timestamp'].dt.hour
    for feature in one_hot_feature:
            #nc = LabelEncoder().fit(data[feature])
            data[feature]= LabelEncoder().fit_transform(data[feature].astype(str))
    print('ok feature')
    
    
    
    
    
    X=data.copy()
    
    
    
    ddd[qqq]=X
    

    
    
ddd['dc'].to_pickle('./dc')
ddd['bc'].to_pickle('./bc')
ddd['cc'].to_pickle('./cc')
ddd['mc'].to_pickle('./mc')  
    
    
for qqq in ddd:
    X=ddd[qqq]
    
    
    
    
    
    
    
    #data['Label'].replace('BENIGN',0)
    #data['Label'].replace('PortScan',1)
    #data['Label'].replace('DDoS',2)
    y=X.pop('Label')
    
    y=y.replace('Normal',0)
    y=y.replace('Attack',1)
    y=y.replace('A ttack',1)
    
    X_train, test, y_train, test_y = train_test_split(X, y, test_size=0.25, random_state=1)
    train, valid, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    print('X Y train test val READY')
    
    '''
    enc = OneHotEncoder()
    train_x=train[cf]
    valid_x=valid[cf]
    test_x=test[cf]
    for feature in one_hot_feature:
    
            enc = OneHotEncoder()
            enc.fit(np.concatenate([train[feature].values.reshape(-1, 1),test[feature].values.reshape(-1, 1),valid[feature].values.reshape(-1, 1)]))
            train_a=enc.transform(train[feature].values.reshape(-1, 1))
            valid_a=enc.transform(valid[feature].values.reshape(-1, 1))
            test_a = enc.transform(test[feature].values.reshape(-1, 1))
            train_x= sparse.hstack((train_x, train_a))
            valid_x= sparse.hstack((valid_x, valid_a))
            test_x = sparse.hstack((test_x, test_a))
            print(feature)
    ''' #no onehot
    train_x = train
    valid_x = valid
    test_x = test
    
    
    #for i in ['Timestamp','LIT301','AIT201','AIT503']:
    for i in ['Timestamp']:
        train_x.pop(i)
        valid_x.pop(i)
        test_x.pop(i)

    clf = clflib.LGBMClassifier(
            boosting_type='gbdt', num_leaves=20, reg_alpha=0.0, reg_lambda=1, objective= 'binary',
                metric= 'binary_logloss',
                num_class=1, verbose=-1, verbose_eval = -1,
            max_depth=4, n_estimators=500,
            subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
            learning_rate=0.1,random_state=2020, n_jobs=-1,num_rounds = 10
        )
    
#    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],early_stopping_rounds=100)
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)])
    #res = clf.predict_proba(test_x)[:,1]
    
    
    pred = clf.predict(test_x)
    
    
    pred1 = 1*(pred==1)
    
    
    y1=1*(test_y==1)
    
    
    
    from sklearn.metrics import accuracy_score
    from sklearn import metrics
    
    
    acc1 = accuracy_score(y1, pred1)
    pre1 = metrics.precision_score(y1, pred1, average='macro')
    rec1 = metrics.recall_score(y1, pred1, average='macro')
    print("%s - ACC: %f PRE: %f REC: %f F1: %f" % (qqq,acc1,pre1,rec1,2*pre1*rec1/((pre1+rec1))))
    
    
#    import matplotlib.pylab as plt
#    plt.figure(figsize=(24,12))
#    lgb.plot_importance(clf, max_num_features=30)
#    plt.title("Featurertances")
#    plt.show()















