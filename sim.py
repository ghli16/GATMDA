import pandas as pd
import numpy as np



def load_data():
    '''
    D_SSM1 = np.loadtxt('data/2.disease semantic similarity 1/disease semantic similarity matrix 1.txt')
    D_SSM2 = np.loadtxt('data/2.disease semantic similarity 1/disease semantic similarity matrix 2.txt')
    D_SSM = (D_SSM1 + D_SSM2) / 2

    M_FSM = np.loadtxt('data/3.miRNA functional simialrity/functional similarity matrix.txt')
    '''

    M_FSM = np.loadtxt('data2/miRNA-miRNA.csv', delimiter=',', dtype=float)
    D_SSM = np.loadtxt('data2/disease-disease.csv', delimiter=',', dtype=float)


    SD = D_SSM
    SM = M_FSM

    return SD, SM