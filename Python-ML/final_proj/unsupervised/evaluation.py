'''
Created on Dec 12, 2012

@author: jason
'''

from util.mlExceptions import *
from collections import Counter
from inspect import stack
import numpy as np
from scipy.linalg import norm
from scipy.stats.distributions import entropy
from gmm_diag2_new import *
import pickle as pkl

def getClusterHistogram(pointsInCluster, trueLabelHash, histSize = 101):
    '''
    Given all the points in a cluster, extract a histogram which plots the 
    amount of those points allocated to a true label of the data. This function
    is used as a degree of approximation to cluster accuracy.
    
    @param pointsInCluster: 2D ndarray which holds all the D-dimensional points assigned to the cluster.
    @param trueLabelHash: a hash table which maps the index of a point to its true label in the data.
    @param histSize: The size of the histogram to build (by default 101, corresponding to Caltech 101's labels).
    @return a histogram as described above
    @return the maximal true label in the cluster
    @raise LogicalError when the first or second parameter is None, or when histSize <= 0.      
    '''
    
    if pointsInCluster is None or trueLabelHash is None:
        raise LogicalError, "Method %s: None arguments were provided." %(stack()[0][3])
    if histSize is None or histSize <=0:
        raise LogicalError, "Method %s: histSize parameter should be a positive integer (provided: %s)" %(stack()[0][3], str(histSize))
    
    trueLabels = [trueLabelHash[point] for point in pointsInCluster]
    histogram, _binEdges = np.histogram(trueLabels, range(0, histSize + 1))
    return histogram, Counter(trueLabels).most_common(1)[0][0]

def evaluateClustering(centroids, data, assignments, trueLabelMeans, trueLabelHash, histSize=101):
    
    '''
    Evaluates a clustering algorithm, when the true labels of the data have been given. Those
    labels are contained as mapped values in "trueLabelHash". 
    
    To evaluate the clustering algorithm's accuracy, we will follow twp base approach. To do this, we first
    observe that it is possible to compute the distance of every centroid to the mean values of the 
    true labels. Therefore, for every cluster it is possible to find the category mean to which it is closest in vector space.: 
    
    Approach #1: We will associate each centroid with its closest label and therefore compute the clustering
    quality in terms of misclassification error. In this case, the predicted labels are the clusters that 
    examples are assigned to.
    
    Approach #2: For every cluster, we build a histogram which plots the distribution of its points over 
    the ***true*** labels. Clusters whose points' majority true label coincide with the label whose mean 
    is closest to the centroid are more "accurate" than ones for which this condition does not hold.
    
    @param centroids: K x D ndarray, representing K centroids in D-space.
    @param data: N x D ndarray, representing the training data X.
    @param assignments: N-sized ndarray, mapping each example to its cluster. assignments[i] = k means that
            the ith example in "data" is mapped to the cluster represented by the kth centroid.
    @param trueLabelMeans: |labels| xD ndarray, holding the D-dimensional mean values of every class
    @param trueLabelHash: A hash which maps example indices to their true label.
    @param histSize: integer which represents the size of the histogram to pass to "getClusterHistogram".
            By default it's equal to 101, the amount of labels in Caltech101.
    @raise LogicalError: For various cases which have to do with argument sanity checking.
    @raise DatasetError: If provided with no data.
    @return The number of "accurate" clusters, as defined above.
    '''
    
    if centroids is None or assignments is None or trueLabelMeans is None or trueLabelHash is None:
        raise LogicalError, "Method %s: \"None\" argument(s) provided." %(stack()[0][3])
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        raise DatasetError, "Method %s: No training data provided." %(stack()[0][3])
    if histSize is None or histSize <= 0:
        raise LogicalError, "Method %s: histSize parameter should be a positive integer (provided: %s)." %(stack()[0][3], str(histSize))
    
    if len(trueLabelMeans) != 101:
        raise LogicalError, "Method %s: trueLabelMeans array should have 101 dimensions." %(stack()[0][3])
    
    # for each centroid, find the category mean it is closest to. Then associate this cluster with this
    # mean in a hash.
     
     
    # I have tried quite a bit to find an efficient solution to this, and have failed. Instead,
    # I will write an inefficient for loop - based implementation.
     
    # Careful: the trueLabelMeans 2D ndarray is zero-indexed, whereas the labels are not!
     
    closestLabel = dict()
    for i in range(len(centroids)):
        closestLabel[i] = np.array([norm(centroids[i] - mean) for mean in trueLabelMeans]).argmin() + 1
       
    
    # Implement approach #1: Assuming that every assigned cluster is a predicted label, compute
    # the cluster accuracy in terms of misclassification error.
    
    misclassifiedPoints = 0
    for exIndex in range(data.shape[0]):
        if trueLabelHash[exIndex] != closestLabel[assignments[exIndex]]:
            misclassifiedPoints+=1
    
    errorRate = 100*(misclassifiedPoints/float(data.shape[0]))
    
    # Implement approach #2: Compute true label count histograms and gauge which clusters are "good". 
    # "Good" clusters are closest to the mean of the majority 
    # vote label voted by their points, as reported by the respective histogram.
    
    goodCentroids = 0
    histogramEntropies = []
    for i in range(len(centroids)):
        # Get the indices of all the points in the cluster
        pointsInCluster = [j for j in range(len(assignments)) if assignments[j] == i]
        if len(pointsInCluster) > 0:
            clusterHist, majVoteLabel = getClusterHistogram(pointsInCluster, trueLabelHash, histSize)
            histogramEntropies.append(entropy([val for val in clusterHist if val > 0]))
            if closestLabel[i] != None and majVoteLabel == closestLabel[i]:
                goodCentroids+=1 
    
    # Return all metrics to caller.
    
    return errorRate, goodCentroids, np.mean(histogramEntropies)


import os

if __name__ == "__main__":
    
    os.chdir('../')   
    gradientData = pkl.load('proc_data/gradient_features_traindat.pkl')
    
    g1 = GMM(101, cvtype='full', thresh = 1e-05)
    g1
    