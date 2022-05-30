"""
Authors: Kat Dover, Anna Ma, Ph.D. and Zixuan Cang, Ph.D.
Principal Investigator: Qing Nie, Ph.D. and Roman Vershynin, Ph.D. from University of California, Irvine
30 May 2022
AVIDA Algorithm: Alternating method for Visualizing and Integration Data
Correspondence: doverk@uci.edu
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

def avida_tsne(X1=np.array([]), X2=np.array([]), alpha=0.0005, no_dims=2, perplexity=30.0,\
                      normalize=0,sample_size=0,visualize=0,labels1=np.array([]),\
                      labels2=np.array([]),graph_dist=False,partial=False,partial_size=1):
    """
        avida_tsne implements the AVIDA algorithm using Gromov-Wasserstein as the optimal transport method
        and t-SNE as the visualization method.
        
        Much of the t-SNE code was built from the original t-SNE implementation by Laurens van der Maaten on 20-12-08.
        A more complete version of t-SNE can be found here (https://lvdmaaten.github.io/tsne/)
        
        INPUTS: X1, X2 in the form of numpy arrays/matrices, where the rows correspond to samples 
        and columns correspond to features.
        
        OUTPUTS: Y1, Y2, in the form of numpy arrays/matrices, where the rows correspond to samples and 
        columns correspond to 2D representation.
        
        PARAMETERS: 
            alpha: indication if optimal transport should be performed. If alpha == 0, t-SNE will be performed 
                    individually on X1, X2 with no influence on each other.
            no_dims: the number of dimensions of the outputs Y1, Y2. Default is 2 (for visualization purposes).
            perplexity: perplexity value for t-SNE, determines the number of nearest neighbors that have
                        a strong effect on the P-values, default set to 30.
            normalize: if data has wide variety of distances, set to 1 to normalize. Otherwise, set to 0.
            sample_size: If greater than 0, algorithm will perform optimal transport on sub sample of data.
            visualize: If visualize is set equal to 1, the algorithm will display Y1 and Y2 each 100 iterations.
            labels1: Class labels for X1.
            labels2: Class labels for X2.
            graph_dist: If set to TRUE, pairwise distances will be based on the graph distances.
            partial: If set to TRUE, partial GW optimal transport will be performed
            partial_size: If Partial is set to TRUE, partial GW optimal transport will be performed 
                            with percentage equal to partial_size.
        
    """
    ## start the timer
    start = timeit.default_timer()
    
    ## Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1    
    
    ## set seed
    np.random.seed(4)
    
    ## normalize distances for complicated data
    if normalize == 1:
        X1 = preprocessing.normalize(X1)
        X2 = preprocessing.normalize(X2)
        
    ## Initialize variables
    (n1, d1) = X1.shape
    (n2, d2) = X2.shape    
    max_iter = 1000
    eta = 500   
    
    Y1 = np.random.randn(n1, no_dims)*1e-3
    Y2 = np.random.randn(n2, no_dims)*1e-3
    Gs = np.random.randn(n1, n2)*0

    dY1 = np.zeros((n1, no_dims))
    dY2 = np.zeros((n2, no_dims))
    a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    
    ## if we are doing a partial transfer, make sure to properly normalize a and b
    if(partial==True):
        a,b = ot.unif(n1),ot.unif(n2)    
    ## if we are sampling the transfer, make sure to 
    if sample_size > 0:
        sample_a, sample_b = np.ones((sample_size,)) / sample_size, np.ones((sample_size,)) / sample_size
        
        
    ## Compute P-values for the first dataset
    P1 = ut.x2p(X1, 1e-5, perplexity)
    P1 = P1 + np.transpose(P1)
    P1 = P1 / np.sum(P1)
    P1 = P1 * 4. ## for early exaggeration
    P1 = np.maximum(P1, 1e-12)
    
    ## Compute P-values for the second dataset
    P2 = ut.x2p(X2, 1e-5, perplexity)
    P2 = P2 + np.transpose(P2)
    P2 = P2 / np.sum(P2)
    P2 = P2 * 4. ## for early exaggeration
    P2 = np.maximum(P2, 1e-12)    
    alphat = 0;   
    
    ## Run iterations
    for iter in range(max_iter):   
        ## Compute pairwise affinities and Q-values for first dataset
        sum_Y1 = np.sum(np.square(Y1), 1)
        num1 = -2. * np.dot(Y1, Y1.T)
        C1 = np.add(np.add(num1, sum_Y1).T, sum_Y1)
        num1 = 1. / (1. + C1)
        num1[range(n1), range(n1)] = 0.
        Q1 = num1 / np.sum(num1)
        Q1= np.maximum(Q1, 1e-12)
        
        ## Compute pairwise affinities and Q-values for second dataset
        sum_Y2 = np.sum(np.square(Y2), 1)
        num2 = -2. * np.dot(Y2, Y2.T)
        C2 = np.add(np.add(num2, sum_Y2).T, sum_Y2)
        num2 = 1. / (1. + C2)
        num2[range(n2), range(n2)] = 0.
        Q2 = num2 / np.sum(num2)
        Q2= np.maximum(Q2, 1e-12)  
        
        ## Compute gradient for t-SNE portion
        PQ1 = P1 - Q1        
        for i in range(n1):
            dY1[i, :] = np.sum(np.tile(PQ1[:, i] * num1[:, i], (no_dims, 1)).T * (Y1[i, :] - Y1), 0)
            
        PQ2 = P2 - Q2
        for i in range(n2):
            dY2[i, :] = np.sum(np.tile(PQ2[:, i] * num2[:, i], (no_dims, 1)).T * (Y2[i, :] - Y2), 0)
            
        ## initialize gradients for GW
        y_reg_grads1 = np.zeros((n1,2))
        y_reg_grads2 = np.zeros((n2,2))
        
        ## Compute the GW and apply if we are our of early exaggeration and on iteration % 100
        if((alphat != 0) & (iter > 200)):
            ## if we are not sampling
            if((sample_size == 0)):
                ## if applicable, compute pairwise graph distances
                if((graph_dist==True)):
                    C1_graph = ut.get_graph_distance_matrix(Y1, int(perplexity), mode="connectivity",\
                                                             metric="correlation") 
                    C2_graph = ut.get_graph_distance_matrix(Y2, int(perplexity), mode="connectivity",\
                                                             metric="correlation")
                    C1_graph /= C1_graph.max()
                    C2_graph /= C2_graph.max()
                    C1_full = C1_graph
                    C2_full = C2_graph
                ## if applicable, compute pairwise euclidean distances
                else:
                    C1_full = sp.spatial.distance.cdist(Y1,Y1)
                    C2_full = sp.spatial.distance.cdist(Y2,Y2)
                ## if we are using partial GW, compute GW
                if((partial == False)):
                    Gs, log = ot.gromov.entropic_gromov_wasserstein(C1_full, C2_full, a, b, 'square_loss',\
                                                                    epsilon=5e-3,max_iter=200,\
                                                                    tol=1e-4,log=True,verbose=True)
                    print(log['gw_dist'])
                    
                ## if we are not using partial GW, compute GW
                else:
                    Gs, log = ot.partial.entropic_partial_gromov_wasserstein(C1_full, C2_full, a,\
                                                                    b,reg=10,m=partial_size,log=True,verbose=True)
                    print(log['partial_gw_dist'])
                    
                ## do thresholding for GW
                for i in range(Gs.shape[0]):
                    tau = np.sort(Gs[i,])[-5]
                    Gs[i,][Gs[i,] < tau] = 0
                    Gs[i,] = Gs[i,]/np.sum(Gs[i,])
                    
                #now update Y1
                Y1 = np.matmul(Gs,Y2)
 
            ## if we are sampling, do the same process
            else:
                ## take our sample
                idx1 = np.random.choice(Y1.shape[0],sample_size,replace=False)
                idx2 = np.random.choice(Y2.shape[0],sample_size,replace=False)
                sample_Y1 = Y1[idx1]
                sample_Y2 = Y2[idx2]
                
                ## find the pairwise graph distances
                if((graph_dist==True)):
                    sample_C1 = ut.get_graph_distance_matrix(sample_Y1, int(perplexity), mode="connectivity",\
                                                          metric="correlation")
                    sample_C2 = ut.get_graph_distance_matrix(sample_Y2, int(perplexity), mode="connectivity",\
                                                          metric="correlation")
                    sample_C1 /= sample_C1.max() 
                    sample_C2 /= sample_C2.max()
                    
                ## find the pairwise euclidean distances
                else:
                    sample_C1 = sp.spatial.distance.cdist(sample_Y1,sample_Y1)
                    sample_C2 = sp.spatial.distance.cdist(sample_Y2,sample_Y2)
                
                ## calculate complete GW
                if((partial == False)):
                    sample_Gs, log0 = ot.gromov.entropic_gromov_wasserstein(sample_C1, sample_C2, sample_a,\
                                                                            sample_b,'square_loss',\
                                                                            epsilon=5e-3,max_iter=200,\
                                                                            tol=1e-4,log=True,verbose=True)
                    print(log0['gw_dist'])
                
                ## calculate partial GW
                else:
                    sample_Gs, log0 = ot.partial.entropic_partial_gromov_wasserstein(sample_C1, sample_C2,\
                                                                                     sample_a,sample_b,reg=10,\
                                                                                     m=partial_size,log=True,\
                                                                                     verbose=True)
                    print(log0['partial_gw_dist'])
                
                ## do thresholding
                for i in range(sample_Gs.shape[0]):
                    tau = np.sort(sample_Gs[i,])[-3]
                    sample_Gs[i,][sample_Gs[i,] < tau] = 0
                    sample_Gs[i,] = sample_Gs[i,]/np.sum(sample_Gs[i,])
                    
                ## now update gradients, loss and update Y1 & Y2
                y_reg_grads1[idx1] = 2 * np.matmul(sample_Gs,sample_Y2) - sample_Y1
                y_reg_grads2[idx2] = 2 * np.matmul(sample_Gs.T,np.matmul(sample_Gs,sample_Y2)-sample_Y1)
                
                sample_Y1 = np.matmul(sample_Gs,sample_Y2)
                Y1[idx1] = sample_Y1

        else:
            ## update Y1, Y2 with gradients
            Y1 = Y1 - eta * dY1 - alphat * 1/n1 * y_reg_grads1 
            Y1 = Y1 - np.tile(np.mean(Y1, 0), (n1, 1))
            
            Y2 = Y2 - eta * dY2 - alphat * 1/n2 * y_reg_grads2
            Y2 = Y2 - np.tile(np.mean(Y2, 0), (n2, 1))
            
        
        ## print out cost function
        if(iter + 1) % 100 == 0:
            C1 = np.sum(P1 * np.log(P1 / Q1))
            C2 = np.sum(P2 * np.log(P2 / Q2))
            print("Iteration %d: error is %f" % (iter + 1, C1+C2))
        
        ## check if upcoming iteration needs to apply GW
        if(iter + 1) % 100 == 0 and iter >= 250:
            alphat = alpha
        else:
            alphat = 0
        
        ## visualize the output
        if(visualize != 0) and ((iter % 50) == 0):
            plt.scatter(Y1[:,0],Y1[:,1],c=labels1)
            plt.scatter(Y2[:,0],Y2[:,1],c=labels2)
            plt.show()
            
        ## stop early exaggeration outside of 200 iterations
        if iter == 200:
            P1 = P1 / 4.
            P2 = P2 / 4.
            
    stop = timeit.default_timer()
    print('Time: ',stop-start)
    
    return Y1, Y2