# coding=UTF-8
import gc
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import random
from GAT import gat

import sim
emb_feature_num=40

from datetime import datetime


def matrix_mul(matrix1,matrix2):
    result = np.matmul(matrix1,matrix2)

    return result

def data_pre(dis_lncRNA_matrix,miRNA_lncRNA_matrix,miRNA_sim_matrix,dis_sim_matrix,miRNA_dis_matrix):

    feature_matrix1 = matrix_mul(miRNA_sim_matrix, miRNA_lncRNA_matrix)
    feature_matrix2 = matrix_mul(dis_sim_matrix, dis_lncRNA_matrix)



    mat1_row = feature_matrix1.shape[0]
    mat1_col = feature_matrix1.shape[1]
    mat2_row = feature_matrix2.shape[0]
    mat2_col = feature_matrix2.shape[1]

    feature_matrix_net = np.empty([mat1_row * mat2_row, 2 * mat1_col], dtype=float)

    feature_matrix = []
    label_vector = []


    as_row = miRNA_dis_matrix.shape[0]
    as_cow = miRNA_dis_matrix.shape[1]
    for m in range(as_row):
        for n in range(as_cow):
            label_vector.append(miRNA_dis_matrix[m, n])

    gat_feature = np.vstack((feature_matrix1,feature_matrix2))


    flag = 0
    for m in range(mat1_row):
        for n in range(mat2_row):
            for l in range(mat1_col):
                feature_matrix_net[flag, l] = feature_matrix1[m, l]
            for l in range(mat2_col):
                feature_matrix_net[flag, mat1_col + l] = feature_matrix2[n, l]
            flag = flag + 1



    feature_matrix_net = np.array(feature_matrix_net)



    feature_matrix = [i for i in feature_matrix_net.tolist()]

    return gat_feature,feature_matrix,label_vector



def cross_validation_experiment(dis_lncRNA_matrix,miRNA_lncRNA_matrix,miRNA_sim_matrix,dis_sim_matrix,miRNA_dis_matrix):

    gat_feature,linear_feature_matrix,label_vector = data_pre(dis_lncRNA_matrix,miRNA_lncRNA_matrix,miRNA_sim_matrix,dis_sim_matrix,miRNA_dis_matrix)

    linear_feature_matrix = np.array(linear_feature_matrix)
    label_vector = np.array(label_vector)



    none_zero_position = np.where(label_vector != 0)

    none_zero_row_index = none_zero_position[0]



    zero_position = np.where(label_vector == 0)
    zero_row_index = zero_position[0]


    positive_randomlist = [i for i in range(len(none_zero_row_index))]

    random.shuffle(positive_randomlist)



    k_folds = 5

    for k in range(k_folds):


        if k != k_folds - 1:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds):(k + 1) * int(
                len(none_zero_row_index) / k_folds)]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))

        else:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds)::]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))

        positive_test_row = none_zero_row_index[positive_test]


        positive_train_row = none_zero_row_index[positive_train]



        train_miRNA_dis_matrix = np.copy(miRNA_dis_matrix)
        train_miRNA_dis_matrix = gat.data_pre(positive_test_row,train_miRNA_dis_matrix)

        test_row = np.append(positive_test_row, zero_row_index)
        train_row = np.append(positive_train_row, zero_row_index)


        miRNA_dis_emb = gat.Get_embedding_Matrix(gat_feature,train_miRNA_dis_matrix)


        miRNA_len = miRNA_dis_matrix.shape[0]
        miRNA_emb_matrix = np.array(miRNA_dis_emb[0:miRNA_len, 0:])
        dis_emb_matrix = np.array(miRNA_dis_emb[miRNA_len::, 0:])


        feature_matrix = gat.append_feature(linear_feature_matrix,miRNA_emb_matrix,dis_emb_matrix,emb_feature_num)


        feature_col = feature_matrix.shape[1]

        train_feature_matrix = np.empty([len(train_row),feature_matrix.shape[1]],dtype=float)
        train_label_vector = []
        test_feature_matrix = np.empty([len(test_row),feature_matrix.shape[1]],dtype=float)


        for l in range(len(train_row)):
            for i in range(feature_col):
                train_feature_matrix[l][i] = feature_matrix[train_row[l]][i]
            train_label_vector = np.append(train_label_vector,label_vector[train_row[l]])

        for l in range(len(test_row)):
            for i in range(feature_col):
                test_feature_matrix[l][i] = feature_matrix[test_row[l]][i]


        train_feature_matrix = np.array(train_feature_matrix,dtype=float)
        train_label_vector = np.array(train_label_vector,dtype=int)
        test_feature_matrix = np.array(test_feature_matrix, dtype=float)
        clf = RandomForestClassifier(n_estimators=350, oob_score=False, n_jobs=-1)
        clf.fit(train_feature_matrix,train_label_vector)



        predict_y_proba = clf.predict_proba(test_feature_matrix)[:, 1]




        del train_feature_matrix
        del train_label_vector
        del test_feature_matrix

        gc.collect()





if __name__ == '__main__':

    datetime1 = datetime.now()
    #读取数据
    disease_lncRNA_matrix = np.loadtxt('data2/lncdis.csv',delimiter=',',dtype=float)
    disease_lncRNA_matrix = disease_lncRNA_matrix.T
    
    miRNA_lncRNA_matrix = np.loadtxt('data2/milnc.csv',delimiter=',',dtype=float)
    disease_sim_matrix,miRNA_sim_matrix = sim.load_data()
    miRNA_disease_matrix = np.loadtxt('data2/midis.csv',delimiter=',',dtype=float)
    

    circle_time = 1


    for i in range(circle_time):
        cross_validation_experiment(disease_lncRNA_matrix,miRNA_lncRNA_matrix,miRNA_sim_matrix,disease_sim_matrix,miRNA_disease_matrix)


