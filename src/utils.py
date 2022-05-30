"""
Authors: Kat Dover, Anna Ma, Ph.D. and Zixuan Cang, Ph.D.
30 May 2022
utils for AVIDA

Citations: get_graph_distance_matrix comes from SCOT repository, given at (https://rsinghlab.github.io/SCOT/) from authors
           Pinar Demetrci and Rebecca Santorella.
           
           Hbeta and x2p are functions to assist the t-SNE portion of AVIDA and the functions come from the t-SNE python
           implementation provided at (https://lvdmaaten.github.io/tsne/) from author Laurens van der Maaten.
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

def get_graph_distance_matrix(data, num_neighbors, mode="connectivity", metric="correlation"):
    """
    Original code from utils file provided in SCOT documentation. Full credit is given to authors Pinar Demetrci and 
    Rebecca Santorella, full code can be found in repository (https://rsinghlab.github.io/SCOT/)
    Compute graph distance matrices on data 
    """
    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    graph_data=kneighbors_graph(data, num_neighbors, mode=mode, metric=metric, include_self=include_self)
    shortestPath_data= dijkstra(csgraph= csr_matrix(graph_data), directed=False, return_predecessors=False)
    shortestPath_max= np.nanmax(shortestPath_data[shortestPath_data != np.inf])
    shortestPath_data[shortestPath_data > shortestPath_max] = shortestPath_max
    shortestPath_data=shortestPath_data/shortestPath_data.max()

    return shortestPath_data

def Hbeta(D=np.array([]), beta=1.0):
    """
        Original code from tsne.py authored by Laurens van der Maaten, full code can be found at github (https://lvdmaaten.github.io/tsne/)
        
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Original code from tsne.py authored by Laurens van der Maaten, full code can be found at github (https://lvdmaaten.github.io/tsne/)
        
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def y2q(Y=np.array([])):
    """
        Finds the Q-values for representation Y. The Q-values are based on the kernel 
        from the student t-distribution
    """
    (n, d) = Y.shape
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    C = np.add(np.add(num, sum_Y).T, sum_Y)
    num = 1. / (1. + C)
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    Q= np.maximum(Q, 1e-12)    
    return Q