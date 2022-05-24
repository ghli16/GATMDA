
import numpy as np
import torch
from GAT import GAT_layer
import networkx as nx
import pandas as pd

def data_pre(positive_test_label,miRNA_disease_matrix):
    for i in positive_test_label:
        y=i%432
        x=int((i-y)/432)
        miRNA_disease_matrix[x][y]=0

    return miRNA_disease_matrix

def constructNet(miRNA_dis_matrix):
    miRNA_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[0], miRNA_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[1], miRNA_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((miRNA_matrix,miRNA_dis_matrix))

    mat2 = np.hstack((miRNA_dis_matrix.T,dis_matrix))


    return np.vstack((mat1,mat2))


def Net2edgelist(miRNA_disease_matrix_net):
    none_zero_position = np.where(np.triu(miRNA_disease_matrix_net) != 0)
    none_zero_row_index = np.mat(none_zero_position[0],dtype=int).T
    none_zero_col_index = np.mat(none_zero_position[1],dtype=int).T
    none_zero_position = np.hstack((none_zero_row_index,none_zero_col_index))

    none_zero_position = np.array(none_zero_position)

    name = 'GAT/miRNA_disease.txt'
    np.savetxt(name, none_zero_position,fmt="%d",delimiter=' ')

def get_embedding(vectors: dict):
    matrix = np.zeros((

        869,
        len(list(vectors.values())[0])
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix


def Get_embedding_Matrix(pre_feature,train_miRNA_dis_matrix):

    miRNA_disease_matrix_net = np.mat(constructNet(train_miRNA_dis_matrix))

    Net2edgelist(miRNA_disease_matrix_net)

    pre_feature = torch.Tensor(pre_feature)
    G = np.loadtxt('GAT/miRNA_disease.txt',dtype=int)

    adj = np.empty([pre_feature.shape[0], pre_feature.shape[0]], dtype=int)

    for i in range(G.shape[0]):
        x = G[i][0]
        y = G[i][1]

        adj[x][y] = 1
        adj[y][x] = 1



    adj = torch.Tensor(adj)



    GAL = GAT_layer.GAT(861,68, 40, 0.2, 0.2, 4)


    matrix = GAL(pre_feature, adj)

    matrix = matrix.detach().numpy()
    return matrix

def append_feature(feature_matrix_net,miRNA_emb_matrix,dis_emb_matrix,emb_num):
    last_feature = np.empty([feature_matrix_net.shape[0], feature_matrix_net.shape[1]+emb_num*2], dtype=float)

    for i in range(feature_matrix_net.shape[0]):
        for j in range(feature_matrix_net.shape[1]):
            last_feature[i][j] = feature_matrix_net[i][j]

    flag = 0
    miRNA_row = miRNA_emb_matrix.shape[0]
    dis_row = dis_emb_matrix.shape[0]
    pre_col = feature_matrix_net.shape[1]


    for m in range(miRNA_row):
        for n in range(dis_row):
            for l in range(miRNA_emb_matrix.shape[1]):
                last_feature[flag, pre_col + l] = miRNA_emb_matrix[m, l]
            for l in range(dis_emb_matrix.shape[1]):
                last_feature[flag, pre_col + dis_emb_matrix.shape[1] + l] = dis_emb_matrix[n, l]
            flag = flag + 1

    return last_feature