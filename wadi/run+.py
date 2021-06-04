# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



ddd={}


    
ddd['dc']=pd.read_pickle("./dc")
ddd['bc']=pd.read_pickle("./bc")
ddd['cc']=pd.read_pickle("./cc")
ddd['mc']=pd.read_pickle("./mc")
    


q1=pd.read_pickle("./dc")
q2=pd.read_pickle("./bc")
q3=pd.read_pickle("./cc")
q1.pop('Label')
q2.pop('Label')


ddd['mmc']=pd.concat([q1,q2,q3], axis=1)
import lightgbm as clflib
res=[]
for qqq in ddd:
    X=ddd[qqq]
    

    y=X.pop('Label')

    
    X_train, test, y_train, test_y = train_test_split(X, y, test_size=0.25, random_state=1)
    train, valid, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    print('X Y train test val READY')
    

    train_x = train
    valid_x = valid
    test_x = test
    

    
    clf = clflib.LGBMClassifier(
        boosting_type='gbdt', num_leaves=10, reg_alpha=0.0, reg_lambda=1, objective= 'binary',
            metric= 'binary_logloss',
            num_class=1, verbose=0, verbose_eval = -1,
        max_depth=2, n_estimators=10,
        subsample=0.5, colsample_bytree=0.6, subsample_freq=1,
        learning_rate=0.095, n_jobs=-1,num_rounds = 12
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
    res.append("%s - ACC: %f PRE: %f REC: %f F1: %f" % (qqq,acc1,pre1,rec1,2*pre1*rec1/((pre1+rec1))))


for r in res:
    print(r)
    
    