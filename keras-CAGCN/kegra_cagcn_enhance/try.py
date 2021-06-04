from __future__ import print_function

from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from layers.graph import GraphConvolution
from utils import *

import time

# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience

# Get data
X, A, y = load_data(dataset=DATASET)
y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask = get_splits(y)


import networkx as nx
G=nx.from_numpy_matrix(np.array(A.toarray()))
dd = nx.centrality.betweenness_centrality(G)
xxx = np.array(list(dd.values()))
c = np.array(xxx)
d=c*2000*2000+1
e=log(d)
f=e*200


