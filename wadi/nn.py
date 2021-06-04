# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder





df0 = pd.read_csv("WADI_14days.csv",header=0)

df1 = pd.read_csv("WADI_attackdata.csv",header=0)



for i in ['Row','Date','Time']:
    df0.pop(i)
    df1.pop(i)

df0['Label']=0
df1['Label']=1

key_df0 = df0.keys().tolist()
key_df1 = df1.keys().tolist()

str1='\\\\WIN-25J4RO10SBF\\LOG_DATA\\SUTD_WADI\\LOG_DATA\\'
for i, v in enumerate(key_df0):
    key_df0[i]=key_df0[i].replace(str1,'')
df0.columns = key_df0

for i, v in enumerate(key_df1):
    key_df1[i]=key_df1[i].replace(str1,'')
df1.columns = key_df1

df = pd.concat( [df0.sample(n=345601, random_state=1), df1], axis=0 )








print('load ok')








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











X=df.copy()
#data['Label'].replace('BENIGN',0)
#data['Label'].replace('PortScan',1)
#data['Label'].replace('DDoS',2)
y=X.pop('Label')



X_train, test, y_train, test_y = train_test_split(X, y, test_size=0.25, random_state=1)
train, valid, train_y, valid_y = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

print('X Y train test val READY')


train_x = train
valid_x = valid
test_x = test


import keras
from keras.models import Sequential
from keras.layers import Dense







model = Sequential()
model.add(Dense(units=16, activation='relu',input_dim = 127))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



history = model.fit(train_x, train_y, epochs=1, verbose=1, validation_data=(valid_x,valid_y))

pred = model.predict(test_x)


pred1 = 1*(pred==1)


y1=1*(test_y==1)


from sklearn.metrics import accuracy_score
from sklearn import metrics


acc1 = accuracy_score(y1, pred1)
pre1 = metrics.precision_score(y1, pred1, average='macro')
rec1 = metrics.recall_score(y1, pred1, average='macro')
print(" - ACC: %f PRE: %f REC: %f F1: %f" % (acc1,pre1,rec1,2*pre1*rec1/((pre1+rec1))))



