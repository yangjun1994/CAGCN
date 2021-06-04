from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence


import pickle as pkl

import sys


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)





def load_data(dataset_str):
    if(dataset_str=='cora2'):
        dataset="cora"
        path="data/"+dataset+"/"
        """Load citation network dataset (cora only for now)"""
        print('From this dir')
        print('Loading {} dataset...'.format(dataset))

        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])

        # build graph
        idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + adj.T.multiply(adj.T > adj)

        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

        return features.todense(), adj, labels
    if(dataset_str=='terrorist_attack'):
        dataset="terrorist_attack"
        path="data/"+dataset+"/"
        print('From this dir')
        print('Loading {} dataset...'.format(dataset))
    
        idx_features_labels = np.genfromtxt("{}{}.nodes".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
    
        # build graph
        idx = np.array(idx_features_labels[:, 0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}_loc.edges".format(path, dataset), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
        # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + adj.T.multiply(adj.T > adj)
    
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    
        return features.todense(), adj, labels
    elif(dataset_str=='x'):
        dataset="x"
        path="data/"+dataset+"/"
        print('From this dir')
        print('Loading {} dataset...'.format(dataset))
    
        idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
        features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        labels = encode_onehot(idx_features_labels[:, -1])
    
        # build graph
        idx = np.array(idx_features_labels[:, 0])
        idx_map = {j: i for i, j in enumerate(idx)}
        edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.dtype(str))
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
        # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + adj.T.multiply(adj.T > adj)
    
        print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))
    
        return features.todense(), adj, labels
    else:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
    
        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)
    
        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended
    
        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    
        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
        # idx_test = test_idx_range.tolist()
        # idx_train = range(len(y))
        # idx_val = range(len(y), len(y)+500)
    
        # train_mask = sample_mask(idx_train, labels.shape[0])
        # val_mask = sample_mask(idx_val, labels.shape[0])
        # test_mask = sample_mask(idx_test, labels.shape[0])
    
        # y_train = np.zeros(labels.shape)
        # y_val = np.zeros(labels.shape)
        # y_test = np.zeros(labels.shape)
        # y_train[train_mask, :] = labels[train_mask, :]
        # y_val[val_mask, :] = labels[val_mask, :]
        # y_test[test_mask, :] = labels[test_mask, :]
    
        # return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
        return features.todense(), adj, labels
def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        # a_norm = adj.dot(d).transpose().dot(d).tocsr()
        a_norm = d.dot(adj).dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj



import networkx as nx

def my_normalize_adj(adj, symmetric,db,t):
    if symmetric:
        #c:中心性
        # print(adj)

        G=nx.from_numpy_matrix(np.array(adj.toarray()))
        if (db=='cora'):
            if (t=='degree'):
# #                dd = nx.centrality.betweenness_centrality(G)
#                 dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('corabc.pkl', 'rb'))
                dd2 = pkl.load(open('coradc.pkl', 'rb'))
#                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
#                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c2*0.1
                c=c=c*(len(c)-1)
            if (t=='degree+betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('corabc.pkl', 'rb'))
                dd2 = pkl.load(open('coradc.pkl', 'rb'))
                # pkl.dump(dd, open('corabc.pkl', 'wb'))
                # pkl.dump(dd2, open('coradc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c+c2
                c=c*50+1
            if (t=='betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('corabc.pkl', 'rb'))
                dd2 = pkl.load(open('coradc.pkl', 'rb'))
                xxx = np.array(list(dd.values()))
                # xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                # c2 = np.array(xxx2)
                # c=c+c2
                c=c*50+1
        if (db=='cora2'):
            if (t=='degree'):
                # dd = nx.centrality.betweenness_centrality(G)
                # dd2 = nx.centrality.degree_centrality(G)
                # pkl.dump(dd2, open('cora2dc.pkl', 'wb'))
                # dd = pkl.load(open('corabc.pkl', 'rb'))
                dd2 = pkl.load(open('cora2dc.pkl', 'rb'))
                # xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
#                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c2*0.1
                c=c=c*(len(c)-1)
            if (t=='degree+betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('corabc.pkl', 'rb'))
                dd2 = pkl.load(open('coradc.pkl', 'rb'))
                # pkl.dump(dd, open('corabc.pkl', 'wb'))
                # pkl.dump(dd2, open('coradc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c+c2
                c=c*50+1
            if (t=='betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('cora2bc.pkl', 'rb'))
                # dd2 = pkl.load(open('coradc.pkl', 'rb'))
                # pkl.dump(dd, open('cora2bc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                # xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                # c2 = np.array(xxx2)
                # c=c+c2
                c=c*1000+1
        if (db=='citeseer'):
            if (t=='degree'):
# #                dd = nx.centrality.betweenness_centrality(G)
#                 dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('citeseerbc.pkl', 'rb'))
                dd2 = pkl.load(open('citeseerdc.pkl', 'rb'))
#                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
#                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c2
                c=c=c*(len(c)-1)
            if (t=='degree+betweenness'):
                dd = nx.centrality.betweenness_centrality(G)
                dd2 = nx.centrality.degree_centrality(G)
#                dd = pkl.load(open('citeseerbc.pkl', 'rb'))
#                dd2 = pkl.load(open('citeseerdc.pkl', 'rb'))
                pkl.dump(dd, open('citeseerbc.pkl', 'wb'))
                pkl.dump(dd2, open('citeseerdc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c+c2
                c=c*50+1
            if (t=='betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('citeseerbc.pkl', 'rb'))
                dd2 = pkl.load(open('citeseerdc.pkl', 'rb'))
                xxx = np.array(list(dd.values()))
                # xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                # c2 = np.array(xxx2)
                # c=c+c2
                c=c*50+1
        if (db=='pubmed'):
            if (t=='degree+betweenness'):
#                dd = nx.centrality.betweenness_centrality(G)
#                dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('pubmedbc.pkl', 'rb'))
                dd2 = pkl.load(open('pubmeddc.pkl', 'rb'))
#                pkl.dump(dd, open('pubmedbc.pkl', 'wb'))
#                pkl.dump(dd2, open('pubmeddc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c+c2
                c=c*1000+1
            if (t=='degree'):
                # dd = nx.centrality.betweenness_centrality(G)
                # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('pubmedbc.pkl', 'rb'))
                dd2 = pkl.load(open('pubmeddc.pkl', 'rb'))
                # xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                # c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c2*0.1
                c=c*(len(c)-1)
            if (t=='betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('pubmedbc.pkl', 'rb'))
                dd2 = pkl.load(open('pubmeddc.pkl', 'rb'))
                # pkl.dump(dd, open('pubmedbc.pkl', 'wb'))
                # pkl.dump(dd2, open('pubmeddc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                # xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                # c2 = np.array(xxx2)
                # c=c+c2
                c=c*1000+1





        if (db=='terrorist_attack'):
            if (t=='degree'):
# #                dd = nx.centrality.betweenness_centrality(G)
#                 dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('terrorist_attackbc.pkl', 'rb'))
                dd2 = pkl.load(open('terrorist_attackdc.pkl', 'rb'))
#                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
#                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c2
                c=c=c*(len(c)-1)
            if (t=='degree+betweenness'):
#                dd = nx.centrality.betweenness_centrality(G)
#                dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('terrorist_attackbc.pkl', 'rb'))
                dd2 = pkl.load(open('terrorist_attackdc.pkl', 'rb'))
#                pkl.dump(dd, open('terrorist_attackbc.pkl', 'wb'))
#                pkl.dump(dd2, open('terrorist_attackdc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                
                c=c*1000
                c=(c+c2*len(c2))*0.1
            if (t=='betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('terrorist_attackbc.pkl', 'rb'))
                dd2 = pkl.load(open('terrorist_attackdc.pkl', 'rb'))
                xxx = np.array(list(dd.values()))
                # xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                # c2 = np.array(xxx2)
                # c=c+c2
                c=c*1000+1


        if (db=='x'):
            if (t=='degree'):
# #                dd = nx.centrality.betweenness_centrality(G)
#                 dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('xbc.pkl', 'rb'))
                dd2 = pkl.load(open('xdc.pkl', 'rb'))
#                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
#                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                c=c2
                c=c=c*(len(c)-1)
            if (t=='degree+betweenness'):
                dd = nx.centrality.betweenness_centrality(G)
                dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('xbc.pkl', 'rb'))
                dd2 = pkl.load(open('xdc.pkl', 'rb'))
#                pkl.dump(dd, open('xbc.pkl', 'wb'))
#                pkl.dump(dd2, open('xdc.pkl', 'wb'))
                xxx = np.array(list(dd.values()))
                xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                c2 = np.array(xxx2)
                
                c=c*1000
                c=(c+c2*len(c2))*0.1
            if (t=='betweenness'):
                # dd = nx.centrality.betweenness_centrality(G)
                # # dd2 = nx.centrality.degree_centrality(G)
                dd = pkl.load(open('xbc.pkl', 'rb'))
                dd2 = pkl.load(open('xdc.pkl', 'rb'))
                xxx = np.array(list(dd.values()))
                # xxx2 = np.array(list(dd2.values()))
                c = np.array(xxx)
                # c=np.log(c)
                # c=c*(len(c)-1)
                # c=c*(len(c)-1)*(len(c)-2)
                # c2 = np.array(xxx2)
                # c=c+c2
                c=c*1000+1


















        print('----------------')
        print(c)
        print('----------------')
        d = sp.diags(np.power(c, -0.5).flatten(), 0)
        a_norm = d.dot(adj).dot(d).tocsr()
    else:
        d = sp.diags(np.power(c, -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm

def my_preprocess_adj(adj, symmetric,db,t):
    adj = adj + sp.eye(adj.shape[0])
    adj = my_normalize_adj(adj, symmetric,db,t)
    return adj







def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y,db):
    if (db=='cora'):
        idx_train = range(200)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    if (db=='cora2'):
        idx_train = range(200)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
    if (db=='citeseer'):
        idx_train = range(120)
        idx_val = range(120, 619)
        idx_test = range(2312, 3326)
    if (db=='pubmed'):
#        idx_train = range(60)
#        idx_val = range(60, 560)
#        idx_test = range(18717, 19716)
        idx_train = range(3000)
        idx_val = range(3000, 7000)
        idx_test = range(7000, 19000)
    if (db=='x'):
        idx_train = range(400)
        idx_val = range(400, 600)
        idx_test = range(600, 877)
    if(db=='terrorist_attack'):
        idx_train = list(range(0,10))+list(range(30,217))+list(range(592,651))+list(range(771,774))+list(range(780,785))+list(range(795,961))
        idx_val = list(range(10,20))+list(range(217,404))+list(range(651,710))+list(range(774,776))+list(range(785,790))+list(range(961,1127))
        idx_test = list(range(20,30))+list(range(404,592))+list(range(710,771))+list(range(776,780))+list(range(790,794))  +list(range(1127,1293))
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

import os
def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):

    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian

from scipy.sparse.linalg import eigs
def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        # largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
        largest_eigval = eigs(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape