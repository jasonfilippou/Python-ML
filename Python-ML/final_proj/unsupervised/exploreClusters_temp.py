'''
Created on Dec 19, 2012

@author: jason
'''
'''
Created on Dec 17, 2012

@author: jason

We apply BIC to find the optimal number of clusters
for the Caltech 101 dataset. Once we do that, we take a peek
inside the clusters and examine the distribution of true labels.
We thus aim to find what the "optimal" number of clusters found
by the clustering algorithm intuitively means. The assumption is that
a more advanced feature representation will yield clusters that group
"similar" images together.

We use K-means and GMM to do this and compare results. For K-means,
we use CIML's BIC algorithm, whereas for GMM, we use a built-in implementation
of BIC offered by the library we use.
'''

from util.mlExceptions import *
from inspect import stack
import pickle as pkl
from scipy.cluster.vq import *
from sklearn.mixture import GMM
import numpy as np

def clusterDataSpec(data, k, algorithm):
    '''
    Cluster the given data into a number of clusters determined by BIC.
    @param data: 2D numpy array holding our data.
    @param algorithm: 
    @raise LogicalError if algorithm is other than "k-means" or "GMM"
    @return The predicted labels (clusters) for every example.
    '''
    
    if algorithm not in ["k-means", "GMM"]:
        raise LogicalError, "Method %s: Clustering is made only through K-means or GMM." %(stack()[0][3])
    
    print "Clustering for k=%d." %(k) 
    if algorithm == "k-means":
        whiten(data)
        codebook, _distortion = kmeans(data, k, 10) # 10 iterations only to make it faster
    else:
        g = GMM(n_components=k,thresh = 1e-05, covariance_type='diag', n_iter=10)
        g.fit(data)
            
    #print "Optimal number of clusters according to BIC: %d." %(optimalK)
    
    # Return predicted labels
    if algorithm == "k-means":
        return vq(data, codebook)[0] # predictions on the same data
    else:
        return g.predict(data) # predictions on the same data
    
def clusterData(data, algorithm):
    '''
    Cluster the given data into a number of clusters determined by BIC.
    @param data: 2D numpy array holding our data.
    @param algorithm: 
    @raise LogicalError if algorithm is other than "k-means" or "GMM"
    @return The predicted labels (clusters) for every example.
    '''
    
    if algorithm not in ["k-means", "GMM"]:
        raise LogicalError, "Method %s: Clustering is made only through K-means or GMM." %(stack()[0][3])
    
    bicList = list()
    allComponents = list()
    for k in range(1, 111, 10):
        print "Clustering for k=%d." %(k) 
        if algorithm == "k-means":
            whiten(data)
            codebook, distortion = kmeans(data, k, 10) # 10 iterations only to make it faster
            bicList.append(distortion + k * np.log(data.shape[1])) # CIML's BIC used
            allComponents.append(codebook)
        else:
            g = GMM(n_components=k,thresh = 1e-05, covariance_type='diag', n_iter=10)
            g.fit(data)
            bicList.append(g.bic(data)) # GMM object's BIC implementation used
            allComponents.append(g) # In this case, we want the GMM object to be inserted so that we can call the appropriate "predict"
            
    print "bic list:" + str(bicList)
    pkl.dump(bicList, open('proc_data/bicList.pkl', 'wb'))
    optimalK = np.argmin(bicList) + 1 
    print "Optimal number of clusters according to BIC: %d." %(optimalK)
    
    # Return predicted labels
    if algorithm == "k-means":
        optimalCodeBook = allComponents[np.argmin(bicList)]
        return vq(data, optimalCodeBook)[0] # predictions on the same data
    else:
        optimalMixture = allComponents[np.argmin(bicList)]
        return optimalMixture.predict(data) # predictions on the same data
    

def examineClusters(predictedLabels, trueLabelHash):
    '''
    Given an assignment of examples to labels (clusters), we build a histogram of 
    true labels inside the clusters. We thus aim to better understand the grouping
    effectuated by the clustering algorithm.
    
    @param predictedLabels: A list of ints, representing predicted labels. One per example.
    @param trueLabelHash: A Python dictionary, mapping example indices to true labels.
    @raise LogicalError if the size of the first list is not the same as the size of the keys of the dictionary.
    @return a list of true label histograms for every cluster.
    '''
    
    if len(predictedLabels) != len(trueLabelHash.keys()):
        raise LogicalError, "Method %s: Mis-match between argument length." %(stack()[0][3])
    
    histogramList = list()
    for cluster in range(np.min(predictedLabels), np.max(predictedLabels) + 1):
        examplesOfCluster = [ex for ex in range(len(predictedLabels)) if predictedLabels[ex] == cluster]
        trueLabels = [trueLabelHash[ex] for ex in examplesOfCluster]
        histogram, _binEdges = np.histogram(trueLabels, range(1, 103)) # All possible Caltech 101 labels considered
        histogramList.append(histogram)
    return histogramList
    
import os

if __name__ == "__main__":
    os.chdir("../")
    
    # Part 1: Gradient Features for both K-means and GMM
    
#    gradientFeatures = pkl.load(open('proc_data/gradient_features_traindat.pkl', 'rb'))
#    gradientExampleHash = pkl.load(open('proc_data/gradient_features_examplehash.pkl', 'rb'))
#    gradientLabelAssociations = pkl.load(open('proc_data/gradientLabelAssociations.pkl', 'rb'))
#   
#    
#    kmeansTwoLabels = clusterDataSpec(gradientFeatures, 2, "k-means")
#
#    kmeansTwoLabHist = examineClusters(kmeansTwoLabels, gradientExampleHash)
#    pkl.dump(kmeansTwoLabHist, open('proc_data/kmeans_twoLabHist.pkl', 'wb'))
#    
#    kmeansThreeLabels = clusterDataSpec(gradientFeatures, 3, "k-means")
#    kmeansThreeLabHist = examineClusters(kmeansThreeLabels, gradientExampleHash)
#    pkl.dump(kmeansThreeLabHist, open('proc_data/kmeans_threeLabHist.pkl', 'wb'))
#    
#    kmeansFiveLabels = clusterDataSpec(gradientFeatures, 5, "k-means")
#    kmeansFiveLabHist = examineClusters(kmeansFiveLabels, gradientExampleHash)
#    pkl.dump(kmeansFiveLabHist, open('proc_data/kmeans_fiveLabHist.pkl', 'wb'))
#    
#    kmeansTenLabels = clusterDataSpec(gradientFeatures, 10, "k-means")
#    kmeansTenLabHist = examineClusters(kmeansTenLabels, gradientExampleHash)
#    pkl.dump(kmeansTenLabHist, open('proc_data/kmeans_tenLabHist.pkl', 'wb'))
    
    #kmeanslabels = pkl.load(open('proc_data/optimalKMeansPredLabels_gradients.pkl', 'rb'))
    #print "Computed predicted labels for K-means on gradient features"
    #pkl.dump(kmeanslabels, open('proc_data/optimalKMeansPredLabels_gradients.pkl', 'wb'))
#    gmmTwoDifflabels = clusterDataSpec(gradientFeatures, 2, "GMM")
#    gmmTwoHist = examineClusters(gmmTwoDifflabels, gradientExampleHash)
#    pkl.dump(gmmTwoHist, open('proc_data/gmmHistForTwoLabels.pkl', 'wb'))
#    
#    print "Done with 1"
#    gmmThreeDiffLabels = clusterDataSpec(gradientFeatures, 3, "GMM")
#    gmmThreeHist = examineClusters(gmmThreeDiffLabels, gradientExampleHash)
#    pkl.dump(gmmThreeHist, open('proc_data/gmmHistForThreeLabels.pkl', 'wb'))
#    print "Done with 2"
#    gmmFiveDiffLabels = clusterDataSpec(gradientFeatures, 5, "GMM")
#    gmmFiveHist = examineClusters(gmmFiveDiffLabels, gradientExampleHash)
#    pkl.dump(gmmFiveHist, open('proc_data/gmmHistForFiveLabels.pkl', 'wb'))
#    print "Done with 3"
#    gmmTenDiffLabels = clusterDataSpec(gradientFeatures, 10, "GMM")
#    gmmTenHist = examineClusters(gmmTenDiffLabels, gradientExampleHash)
#    pkl.dump(gmmTenHist, open('proc_data/gmmHistForTenLabels.pkl', 'wb'))
#    print "Done with 4"
    #gmmlabels = pkl.load(open('proc_data/optimalGMMPredLabels_gradients.pkl'))
    #pkl.dump(gmmlabels, open('proc_data/optimalGMMPredLabels_gradients.pkl', 'wb'))
    #print "Computed predicted labels for GMM on gradient features."
    #kmeansHist = examineClusters(kmeanslabels, gradientExampleHash)
    #print kmeansHist
    #gmmHist = examineClusters(gmmlabels, gradientExampleHash)
    #print sum(gmmHist, 0)
    #print gmmHist
    #print len(gmmHist)
    #pkl.dump(gmmHist, open('proc_data/gradientGMMHistForOptimalK.pkl', 'wb'))
    #quit()
    #print gmmHist
    
    # Part 2: SIFT features for both K-means and GMM
    
    SIFTFeatures = pkl.load(open('proc_data/sift_data_parsed.pkl', 'rb'))
    SIFTExampleHash = pkl.load(open('proc_data/sift_data_exampleHash.pkl', 'rb'))
    SIFTLabelAssociations = pkl.load(open('proc_data/SIFTLabelAssociations.pkl', 'rb'))
    #kmeanslabels = clusterData(SIFTFeatures, "k-means")
    #kmeanslabels = pkl.load(open('proc_data/optimalKMeansPredLabels_SIFT.pkl', 'rb'))
    #print "Computed predicted labels for K-means on SIFT features."
    #pkl.dump(kmeanslabels, open('proc_data/optimalKMeansPredLabels_SIFT.pkl', 'wb'))
    gmmlabels = clusterData(SIFTFeatures, "GMM")
    #print "Computed predicted labels for GMM on SIFT features."
    #pkl.dump(gmmlabels, open('proc_data/optimalGMMPredLabels_SIFT.pkl', 'wb'))
    #kmeansHist = examineClusters(kmeanslabels, SIFTExampleHash)
    #print kmeansHist
    gmmHist = examineClusters(gmmlabels, SIFTExampleHash)
    pkl.dump(gmmHist, open('proc_data/optimalGMMHist_SIFT.pkl', 'wb'))               
    #print kmeansHist
    #print gmmHist
    print "Done."