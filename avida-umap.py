import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise_distances
from sklearn import preprocessing
import random
import matplotlib.pyplot as plt
import ot
import scipy as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
import timeit
import umap

def get_graph_distance_matrix(data, num_neighbors, mode="connectivity", metric="correlation"):
	"""
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
    (n, d) = Y.shape
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)
    C = np.add(np.add(num, sum_Y).T, sum_Y)
    num = 1. / (1. + C)
    num[range(n), range(n)] = 0.
    Q = num / np.sum(num)
    Q= np.maximum(Q, 1e-12)    
    return Q

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y
def test_transfer_accuracy(data1, data2, type1, type2):
    """
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
def integrate_umap_gw(X1=np.array([]), X2=np.array([]), alpha=0.0005, no_dims=2, initial_dims=50,n_neighbors=30,\
                      min_dist=0.1,spread=1.0,repulsion_strength=1.0,\
                      perplexity=30.0,normalize=0,sample_size=0,learning_rate=1.0,negative_sample_rate=5,visualize=0,labels1=np.array([]),\
                      labels2=np.array([]),graph_dist=False,partial=False,partial_size=1,our_log=False):
    #start the timer
    start = timeit.default_timer()
    
    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1    
    
    #set seed
    np.random.seed(4)
    #normalize distances for complicated data
    if normalize == 1:
        X1 = preprocessing.normalize(X1)
        X2 = preprocessing.normalize(X2)
        
    # Initialize variables
    (n1, d1) = X1.shape
    (n2, d2) = X2.shape    
    max_iter = 9
    eta = 500   
    
    tsne_loss_iter = []
    tsne_loss_val = []
    gw_loss_iter = []
    gw_loss_val = []
    alignment_iter = []
    alignment_val = []
    accuracy_iter = []
    accuracy_val = []
    vis_iter = []
    vis_val = []
    
    Y1 = np.random.randn(n1, no_dims)*1e-3
    Y2 = np.random.randn(n2, no_dims)*1e-3
    Gs = np.random.randn(n1, n2)*0
    #sample_Gs = np.random.randn(sample_size,sample_size)*0
    dY1 = np.zeros((n1, no_dims))
    dY2 = np.zeros((n2, no_dims))
    a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2
    if(partial==True):
        a,b = ot.unif(n1),ot.unif(n2)    
    if sample_size > 0:
        sample_a, sample_b = np.ones((sample_size,)) / sample_size, np.ones((sample_size,)) / sample_size
        
        
#     # Compute P-values
#     P1 = x2p(X1, 1e-5, perplexity)
#     P1 = P1 + np.transpose(P1)
#     P1 = P1 / np.sum(P1)
#     P1 = P1 * 4.									# early exaggeration
#     P1 = np.maximum(P1, 1e-12)
    
#     P2 = x2p(X2, 1e-5, perplexity)
#     P2 = P2 + np.transpose(P2)
#     P2 = P2 / np.sum(P2)
#     P2 = P2 * 4.									# early exaggeration
#     P2 = np.maximum(P2, 1e-12)    
#     alphat = 0;   
    
    # Run iterations
    for iter in range(max_iter):
        if iter == 0:
            reducer = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,spread=spread,\
                                learning_rate=learning_rate,negative_sample_rate=negative_sample_rate,\
                                repulsion_strength=repulsion_strength,n_epochs=250)
            Y1 = reducer.fit_transform(X1)
            Y2 = reducer.fit_transform(X2)
        else:
            reducer_1 = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,spread=spread,learning_rate=learning_rate,\
                                  negative_sample_rate=negative_sample_rate,repulsion_strength=repulsion_strength,init=Y1,n_epochs=100)
            Y1 = reducer_1.fit_transform(X1)
            reducer_2 = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,spread=spread,learning_rate=learning_rate,\
                                  negative_sample_rate=negative_sample_rate,repulsion_strength=repulsion_strength,init=Y2,n_epochs=100)
            Y2 = reducer_2.fit_transform(X2)
        if(visualize != 0) and ((iter % 1) == 0):
            plt.scatter(Y1[:,0],Y1[:,1],c=labels1)
            plt.scatter(Y2[:,0],Y2[:,1],c=labels2)
            plt.show()
#         # Compute pairwise affinities
#         sum_Y1 = np.sum(np.square(Y1), 1)
#         num1 = -2. * np.dot(Y1, Y1.T)
#         C1 = np.add(np.add(num1, sum_Y1).T, sum_Y1)
#         num1 = 1. / (1. + C1)
#         num1[range(n1), range(n1)] = 0.
#         Q1 = num1 / np.sum(num1)
#         Q1= np.maximum(Q1, 1e-12)
        
#         sum_Y2 = np.sum(np.square(Y2), 1)
#         num2 = -2. * np.dot(Y2, Y2.T)
#         C2 = np.add(np.add(num2, sum_Y2).T, sum_Y2)
#         num2 = 1. / (1. + C2)
#         num2[range(n2), range(n2)] = 0.
#         Q2 = num2 / np.sum(num2)
#         Q2= np.maximum(Q2, 1e-12)  
        
#         # Compute gradient
#         PQ1 = P1 - Q1        
#         for i in range(n1):
#             dY1[i, :] = np.sum(np.tile(PQ1[:, i] * num1[:, i], (no_dims, 1)).T * (Y1[i, :] - Y1), 0)
            
#         PQ2 = P2 - Q2
#         for i in range(n2):
#             dY2[i, :] = np.sum(np.tile(PQ2[:, i] * num2[:, i], (no_dims, 1)).T * (Y2[i, :] - Y2), 0)
            
#         #initialize regularizer gradients
#         y_reg_grads1 = np.zeros((n1,2))
#         y_reg_grads2 = np.zeros((n2,2))
        
        # Compute the regularizer & apply the regularizer
        if(alpha != 0):
            if((sample_size == 0)):
                if((graph_dist==True)):
                    C1_graph = get_graph_distance_matrix(Y1, int(perplexity), mode="connectivity",\
                                                             metric="correlation") 
                    C2_graph = get_graph_distance_matrix(Y2, int(perplexity), mode="connectivity",\
                                                             metric="correlation")
                    C1_graph /= C1_graph.max()
                    C2_graph /= C2_graph.max()
                    C1_full = C1_graph
                    C2_full = C2_graph
                else:
                    C1_full = sp.spatial.distance.cdist(Y1,Y1)
                    C2_full = sp.spatial.distance.cdist(Y2,Y2)
                if((partial == False)):
                    Gs, log = ot.gromov.entropic_gromov_wasserstein(C1_full, C2_full, a, b, 'square_loss',\
                                                                    epsilon=5e-3,max_iter=200,\
                                                                    tol=1e-4,log=True,verbose=True)
                    gw_loss_val = np.append(gw_loss_val,[log['gw_dist']])
                    gw_loss_iter = np.append(gw_loss_iter,[iter+1])
                    print(log['gw_dist'])
                else:
                    Gs, log = ot.partial.entropic_partial_gromov_wasserstein(C1_full, C2_full, a,\
                                                                    b,reg=10,m=partial_size,log=True,verbose=True)
                    gw_loss_val = np.append(gw_loss_val,[log['partial_gw_dist']])
                    gw_loss_iter = np.append(gw_loss_iter,[iter+1])
                    print(log['partial_gw_dist'])
                for i in range(Gs.shape[0]):
                    tau = np.sort(Gs[i,])[-5]
                    Gs[i,][Gs[i,] < tau] = 0
                    Gs[i,] = Gs[i,]/np.sum(Gs[i,])
                #now update Y1
                Y1 = np.matmul(Gs,Y2)
                #fracs = evals.calc_domainAveraged_FOSCTTM(Y1, Y2)
                #alignment_val = np.append(alignment_val,[np.mean(fracs)])
                alignment_val = np.append(alignment_val,[test_alignment_score(Y1,Y2)])
                alignment_iter = np.append(alignment_iter,[iter+1])
                #y_reg_grads1 = 2 * np.matmul(Gs,Y2) - Y1
                #y_reg_grads2 = 2 * np.matmul(Gs.T,np.matmul(Gs,Y2)-Y1)
                
                #Y1 = Y1 - alphat * 1/n1 * y_reg_grads1
                #Y1 = Y1 - np.tile(np.mean(Y1, 0), (n1, 1))
                #Y2 = Y2 - alphat * 1/n2 * y_reg_grads2
                #Y2 = Y2 - np.tile(np.mean(Y2, 0), (n2, 1))
            else:
                idx1 = np.random.choice(Y1.shape[0],sample_size,replace=False)
                idx2 = np.random.choice(Y2.shape[0],sample_size,replace=False)
                sample_Y1 = Y1[idx1]
                sample_Y2 = Y2[idx2]
                if((graph_dist==True)):
                    sample_C1 = get_graph_distance_matrix(sample_Y1, int(perplexity), mode="connectivity",\
                                                          metric="correlation")
                    sample_C2 = get_graph_distance_matrix(sample_Y2, int(perplexity), mode="connectivity",\
                                                          metric="correlation")
                    sample_C1 /= sample_C1.max() 
                    sample_C2 /= sample_C2.max()
                else:
                    sample_C1 = sp.spatial.distance.cdist(sample_Y1,sample_Y1)
                    sample_C2 = sp.spatial.distance.cdist(sample_Y2,sample_Y2)
                if((partial == False)):
                    sample_Gs, log0 = ot.gromov.entropic_gromov_wasserstein(sample_C1, sample_C2, sample_a,\
                                                                            sample_b,'square_loss',\
                                                                            epsilon=5e-3,max_iter=200,\
                                                                            tol=1e-4,log=True,verbose=True)
                    gw_loss_val = np.append(gw_loss_val,[log0['gw_dist']])
                    gw_loss_iter = np.append(gw_loss_iter,[iter+1])
                    print(log0['gw_dist'])
                else:
                    sample_Gs, log0 = ot.partial.entropic_partial_gromov_wasserstein(sample_C1, sample_C2,\
                                                                                     sample_a,sample_b,reg=10,\
                                                                                     m=partial_size,log=True,\
                                                                                     verbose=True)
                    gw_loss_val = np.append(gw_loss_val,[log0['partial_gw_dist']])
                    gw_loss_iter = np.append(gw_loss_iter,[iter+1])
                    print(log0['partial_gw_dist'])
                    
                for i in range(sample_Gs.shape[0]):
                    tau = np.sort(sample_Gs[i,])[-5]
                    sample_Gs[i,][sample_Gs[i,] < tau] = 0
                    sample_Gs[i,] = sample_Gs[i,]/np.sum(sample_Gs[i,])
                    
                #now update gradients and update Y1 & Y2
                y_reg_grads1[idx1] = 2 * np.matmul(sample_Gs,sample_Y2) - sample_Y1
                y_reg_grads2[idx2] = 2 * np.matmul(sample_Gs.T,np.matmul(sample_Gs,sample_Y2)-sample_Y1)
                
                sample_Y1 = np.matmul(sample_Gs,sample_Y2)
                Y1[idx1] = sample_Y1
                #fracs = evals.calc_domainAveraged_FOSCTTM(Y1, Y2)
                #alignment_val = np.append(alignment_val,[np.mean(fracs)])
                alignment_val = np.append(alignment_val,[test_alignment_score(Y1,Y2)])
                alignment_iter = np.append(alignment_iter,[iter+1])
                #Y1 = Y1 - alphat * 1/n1 * y_reg_grads1
                #Y1 = Y1 - np.tile(np.mean(Y1, 0), (n1, 1))
                #Y2 = Y2 - alphat * 1/n2 * y_reg_grads2
                #Y2 = Y2 - np.tile(np.mean(Y2, 0), (n2, 1))
#         else:
#             Y1 = Y1 - eta * dY1 - alphat * 1/n1 * y_reg_grads1 
#             Y1 = Y1 - np.tile(np.mean(Y1, 0), (n1, 1))
            
#             Y2 = Y2 - eta * dY2 - alphat * 1/n2 * y_reg_grads2
#             Y2 = Y2 - np.tile(np.mean(Y2, 0), (n2, 1))
            
        
        #print out cost function
        
#         if(iter + 1) % 1 == 0:
#             C1 = np.sum(P1 * np.log(P1 / Q1))
#             C2 = np.sum(P2 * np.log(P2 / Q2))
#             print("Iteration %d: error is %f" % (iter + 1, C1+C2))
#             #tsne_loss_val = np.append(tsne_loss_val,[C1+C2])
#             #tsne_loss_iter = np.append(tsne_loss_iter,[iter+1])
            
#             #accuracy_iter = np.append(accuracy_iter,[iter+1])
#             #accuracy_val = np.append(accuracy_val,[test_transfer_accuracy(Y1,Y2,labels1,labels2)])
            
#             vis_iter = np.append(vis_iter,[iter+1])
#             min_n = min(n1,n2)
#             newQ1 = y2q(Y1[0:min_n,:])
#             newQ2 = y2q(Y2[0:min_n,:])
            
#             vis_val1 = np.sum(newQ1 * np.log(newQ1/newQ2))
#             vis_val2 = np.sum(newQ2 * np.log(newQ2/newQ1))
#             new_vis_val = 0.5*(vis_val1+vis_val2)
#             vis_val = np.append(vis_val,[new_vis_val])
        
        #update alphat term
#         if(iter + 1) % 100 == 0 and iter >= 250:
#             alphat = alpha
#         else:
#             alphat = 0
        
        #visualize the output
#         if(iter==(max_iter-1)):
#             reducer_1 = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,spread=spread,learning_rate=learning_rate,\
#                                   repulsion_strength=repulsion_strength,init=Y1,n_epochs=50)
#             Y1 = reducer_1.fit_transform(X1)
#             reducer_2 = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,spread=spread,learning_rate=learning_rate,\
#                                   repulsion_strength=repulsion_strength,init=Y2,n_epochs=50)
#             Y2 = reducer_2.fit_transform(X2)
        if(visualize != 0) and ((iter % 1) == 0):
            plt.scatter(Y1[:,0],Y1[:,1],c=labels1)
            plt.scatter(Y2[:,0],Y2[:,1],c=labels2)
            plt.show()
            
#         #stop lying about P-values
#         if iter == 200:
#             P1 = P1 / 4.
#             P2 = P2 / 4.
            
    stop = timeit.default_timer()
    print('Time: ',stop-start)
    
    # Return solution
#     tsne_loss = np.array([tsne_loss_iter,tsne_loss_val])
#     gw_loss = np.array([gw_loss_iter,gw_loss_val])
#     alignment_log = np.array([alignment_iter,alignment_val])
#     accuracy = np.array([accuracy_iter,accuracy_val])
#     vis_loss = np.array([vis_iter,vis_val])
    if(our_log):
        return Y1, Y2, vis_loss
    else:
        return Y1, Y2