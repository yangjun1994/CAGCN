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


cf=['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201','DPIT301','FIT301','LIT301','AIT401','AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504','PIT501','PIT502','PIT503','FIT601']

#one_hot_feature=['MV101','P101','P102','MV201','P201','P202','P203','P204','P205','P206','MV301','MV302','MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603', 'hour']

one_hot_feature=['MV101','P101','P102','MV201','P201','P202','P203','P204','P205','P206','MV301','MV302','MV303','MV304','P301','P302','P401','P402','P403','P404','UV401','P501','P502','P601','P602','P603'] #del hour



data=df
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
#data['hour'] = data['Timestamp'].dt.hour
for feature in one_hot_feature:
        #nc = LabelEncoder().fit(data[feature])
        data[feature]= LabelEncoder().fit_transform(data[feature].astype(str))
print('ok feature')





X=data.copy()

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





import keras
from keras.models import Sequential
from keras.layers import Dense



from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



#model = Sequential()
#model.add(Dense(units=8, activation='relu',input_dim = 51))
#model.add(Dense(units=16, activation='relu'))
#model.add(Dense(units=8, activation='relu'))
#model.add(Dense(units=1, activation='sigmoid'))





model = Sequential()
model.add(Dense(units=16, activation='relu',input_dim = 51))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))




from sklearn.utils import class_weight


class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(train_y),
                                                 train_y)







#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m,precision_m, recall_m])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_m,precision_m, recall_m])

# fit the model
history = model.fit(train_x, train_y, epochs=50, verbose=1,class_weight=class_weights, validation_data=(valid_x,valid_y))

# evaluate the model
loss, f1_score, precision, recall = model.evaluate(test_x, test_y, verbose=1)


print(f1_score, precision, recall)









#pred = model.predict(test_x)
#
#
#pred1 = 1*(pred==1)
#
#
#y1=1*(test_y==1)
#
#
#
#from sklearn.metrics import accuracy_score
#from sklearn import metrics
#
#
#acc1 = accuracy_score(y1, pred1)
#pre1 = metrics.precision_score(y1, pred1, average='macro')
#rec1 = metrics.recall_score(y1, pred1, average='macro')
#print("1-ACC: %f PRE: %f REC: %f F1: %f" % (acc1,pre1,rec1,2*pre1*rec1/((pre1+rec1))))



