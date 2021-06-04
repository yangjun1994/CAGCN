# -*- coding: utf-8 -*-

import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def normalized_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj




def sparse_to_tuple(mx):
    mx = mx.tocoo()
    coords = np.vstack((mx.row, mx.col)).transpose()
    L = tf.SparseTensor(coords, mx.data, mx.shape)
    return tf.sparse_reorder(L)

def calculate_laplacian(adj, lambda_max=1):
    adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    # adj = my_normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)
def my_calculate_laplacian(adj, lambda_max=1):
    # adj = normalized_adj(adj + sp.eye(adj.shape[0]))
    adj = my_normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sp.csr_matrix(adj)
    adj = adj.astype(np.float32)
    return sparse_to_tuple(adj)

def weight_variable_glorot(input_dim, output_dim, name=""):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                            maxval=init_range, dtype=tf.float32)

    return tf.Variable(initial,name=name)



import networkx as nx



def my_normalize_adj(adj):
    G=nx.from_numpy_matrix(np.array(adj))
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))


    # G=nx.from_numpy_matrix(np.array(adj))

    dd = nx.centrality.betweenness_centrality(G)
    dd2 = nx.centrality.degree_centrality(G)
    # dd = pkl.load(open('corabc.pkl', 'rb'))
    # dd2 = pkl.load(open('coradc.pkl', 'rb'))
    # pkl.dump(dd, open('corabc.pkl', 'wb'))
    # pkl.dump(dd2, open('coradc.pkl', 'wb'))
    xxx = np.array(list(dd.values()))
    xxx2 = np.array(list(dd2.values()))
    c = np.array(xxx)
    # c=np.log(c)
    # c=c*(len(c)-1)
    # c=c*(len(c)-1)*(len(c)-2)
    c2 = np.array(xxx2)
    c2=c2*(len(c2)-1)
    # c=c+c2
    # c=c*5+c2
    c=c2
    # c=c2*(len(c2)-1)




    print(c)















    d_inv_sqrt = np.power(c, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    normalized_adj = normalized_adj.astype(np.float32)
    return normalized_adj



# def my_normalize_adj(adj):
#     if True:
#         #c:中心性
#         # print(adj)

#         G=nx.from_numpy_matrix(np.array(adj))

#         dd = nx.centrality.betweenness_centrality(G)
#         dd2 = nx.centrality.degree_centrality(G)
#         # dd = pkl.load(open('corabc.pkl', 'rb'))
#         # dd2 = pkl.load(open('coradc.pkl', 'rb'))
#         # pkl.dump(dd, open('corabc.pkl', 'wb'))
#         # pkl.dump(dd2, open('coradc.pkl', 'wb'))
#         xxx = np.array(list(dd.values()))
#         xxx2 = np.array(list(dd2.values()))
#         c = np.array(xxx)
#         # c=np.log(c)
#         # c=c*(len(c)-1)
#         # c=c*(len(c)-1)*(len(c)-2)
#         c2 = np.array(xxx2)
#         c=c+c2
#         c=c*50+1











#         print('----------------')
#         print(c)
#         print('----------------')
#         d = sp.diags(np.power(c, -0.5).flatten(), 0)
#         a_norm = d.dot(adj).dot(d)
#     else:
#         d = sp.diags(np.power(c, -1).flatten(), 0)
#         a_norm = d.dot(adj).tocsr()
#     return a_norm


