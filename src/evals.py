"""
Authors: Kat Dover, Anna Ma, Ph.D. and Zixuan Cang, Ph.D.
30 May 2022
evals for AVIDA

Citations: test_transfer_accuracy, test_alignment_score, calc_domainAveraged_FOSCTTM and calc_frac_idx are given by the eval.py 
           file from Pamona code. It is authored by Kai Cao and full documentation can be found at
           (https://github.com/caokai1073/Pamona)
"""

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
import random
import ot
import scipy as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import timeit
import utils as ut

def test_transfer_accuracy(data1, data2, type1, type2):
    """
    Author Kai Cao and full documentation can be found at (https://github.com/caokai1073/Pamona)
    
    Metric from UnionCom: "Label Transfer Accuracy"
    """
    Min = np.minimum(len(data1), len(data2))
    k = np.maximum(10, (len(data1) + len(data2))*0.01)
    k = k.astype(np.int)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(data2, type2)
    type1_predict = knn.predict(data1)
    # np.savetxt("type1_predict.txt", type1_predict)
    count = 0
    for label1, label2 in zip(type1_predict, type1):
        if label1 == label2:
            count += 1
    return count / len(type1)

def test_alignment_score(data1_shared, data2_shared, data1_specific=None, data2_specific=None):
    """
        Author Kai Cao and full documentation can be found at (https://github.com/caokai1073/Pamona)
    """
    N = 2

    if len(data1_shared) < len(data2_shared):
        data1 = data1_shared
        data2 = data2_shared
    else:
        data2 = data1_shared
        data1 = data2_shared
    data2 = data2[random.sample(range(len(data2)), len(data1))]
    k = np.maximum(10, (len(data1) + len(data2))*0.01)
    k = k.astype(np.int)

    data = np.vstack((data1, data2))

    bar_x1 = 0
    for i in range(len(data1)):
        diffMat = data1[i] - data
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k+1]
        for j in NearestN:
            if j < len(data1):
                bar_x1 += 1
    bar_x1 = bar_x1 / len(data1)

    bar_x2 = 0
    for i in range(len(data2)):
        diffMat = data2[i] - data
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        NearestN = np.argsort(sqDistances)[1:k+1]
        for j in NearestN:
            if j >= len(data1):
                bar_x2 += 1
    bar_x2 = bar_x2 / len(data2)

    bar_x = (bar_x1 + bar_x2) / 2

    score = 0
    score += 1 - (bar_x - k/N) / (k - k/N)

    data_specific = None
    flag = 0
    if data1_specific is not None:
        data_specific = data1_specific
        if data2_specific is not None:
            data_specific = np.vstack((data_specific, data2_specific))
            flag=1
    else:
        if data2_specific is not None:
            data_specific = data2_specific

    if data_specific is None:
        return score
    else:
        bar_specific1 = 0
        bar_specific2 = 0
        data = np.vstack((data, data_specific))
        if flag==0: # only one of data1_specific and data2_specific is not None
            for i in range(len(data_specific)):
                diffMat = data_specific[i] - data
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k+1]
                for j in NearestN:
                    if j > (len(data1)+len(data2)):
                        bar_specific1 += 1
            bar_specific = bar_specific1
            
        else: # both data1_specific and data2_specific are not None
            for i in range(len(data1_specific)):
                diffMat = data1_specific[i] - data
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k+1]
                for j in NearestN:
                    if j > (len(data1)+len(data2)) and j < (len(data1)+len(data2)+len(data1_specific)):
                        bar_specific1 += 1
       
            for i in range(len(data2_specific)):
                diffMat = data2_specific[i] - data
                sqDiffMat = diffMat**2
                sqDistances = sqDiffMat.sum(axis=1)
                NearestN = np.argsort(sqDistances)[1:k+1]
                for j in NearestN:
                    if j > (len(data1)+len(data2)+len(data1_specific)):
                        bar_specific2 += 1
    
            bar_specific = bar_specific1 + bar_specific2

        bar_specific = bar_specific / len(data_specific)

        score += (bar_specific - k/N) / (k - k/N)

        return score / 2

def vis_loss(X1,X2,Y1,Y2,normalize): 
    """
        Given high-dimensional data X1 and X2 and their low-dimensional representations Y1 and Y2,
        vis_loss generates the P-values for X1 and X2 and Q-Values for Y1 and Y2 and finds the 
        KL-Divergence bewteen X1 and Y1 and X2 and Y2.
    """
    perplexity = 30.0
    
    if normalize == 1:
        X1 = preprocessing.normalize(X1)
        X2 = preprocessing.normalize(X2)
    
    P1 = ut.x2p(X1, 1e-5, perplexity)
    P1 = P1 + np.transpose(P1)
    P1 = P1 / np.sum(P1)
    P1 = np.maximum(P1, 1e-12)
    
    P2 = ut.x2p(X2, 1e-5, perplexity)
    P2 = P2 + np.transpose(P2)
    P2 = P2 / np.sum(P2)
    P2 = np.maximum(P2, 1e-12)    
    
    Q1 = ut.y2q(Y1)
    Q2 = ut.y2q(Y2)
    
    vis_val1 = np.sum(P1 * np.log(P1/Q1))
    vis_val2 = np.sum(P2 * np.log(P2/Q2))
    vis_val = 0.5*(vis_val1+vis_val2)
    return vis_val

def integration_score(Y1,Y2):
    """
        integration_score takes in representations Y1 and Y2 and returns the itegration score,
        or the smallest average distance between the two datasets.
    """
    (n1,d1) = Y1.shape
    (n2,d2) = Y2.shape
    D1 = sp.spatial.distance_matrix(Y1,Y2)
    D2 = sp.spatial.distance_matrix(Y2,Y2)
    val_1 = sum(D1.min(axis=1))/n1
    val_2 = sum(D2.min(axis=1))/n2
    return val_1+val_2


def our_accuracy(Y1,Y2):
    """
        our_accuracy takes in two datasets Y1 and Y2 and returns the sum of the distances of each row.
    """
    (n,d) = Y1.shape
    D = sp.spatial.distance_matrix(Y1,Y2)
    accuracy = (np.trace(D))/n
    return accuracy

def calc_frac_idx(x1_mat,x2_mat):
    """
    Author Kai Cao and full documentation can be found at (https://github.com/caokai1073/Pamona)
    
    Returns fraction closer than true match for each sample (as an array)
    """
    fracs = []
    x = []
    nsamp = x1_mat.shape[0]
    rank=0
    for row_idx in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
        true_nbr = euc_dist[row_idx]
        sort_euc_dist = sorted(euc_dist)
        rank =sort_euc_dist.index(true_nbr)
        frac = float(rank)/(nsamp -1)

        fracs.append(frac)
        x.append(row_idx+1)

    return fracs,x

def calc_domainAveraged_FOSCTTM(x1_mat, x2_mat):
    """
    Author Kai Cao and full documentation can be found at (https://github.com/caokai1073/Pamona)
    
    Metric from SCOT: "FOSCTTM"
    Outputs average FOSCTTM measure (averaged over both domains)
    Get the fraction matched for all data points in both directions
    Averages the fractions in both directions for each data point
    """
    fracs1,xs = calc_frac_idx(x1_mat, x2_mat)
    fracs2,xs = calc_frac_idx(x2_mat, x1_mat)
    fracs = []
    for i in range(len(fracs1)):
        fracs.append((fracs1[i]+fracs2[i])/2)  
    return fracs

