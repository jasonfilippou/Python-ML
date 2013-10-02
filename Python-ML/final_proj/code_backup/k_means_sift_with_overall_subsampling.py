from __future__ import division 
'''
Created on Dec 3, 2012

@author: jason
'''
from util.mlExceptions import *
import os
from inspect import stack
from scipy.io import loadmat
import pickle as pkl
from scipy.cluster.vq import *
from collections import Counter
from numpy.linalg import norm
import numpy as np

# Global variable to hold the current method's name.

def getNonEmptyClusters(assignments):
    
    """
        Given an array of clustering assignments of points, compute and return the unique clusters
    encountered. During clustering, it is possible that the clusters found might not map 1-to-1 with the
    original data labels, especially in the SIFT feature parsing of Caltech, where we sub-sample
    our data: We must therefore use this function to find which clusters exactly were computed.
    @param assignments a 1D numpy array of assignments
    @raise LogicalError if provided with no assignments
    @return a list with integers representing the clusters which had assignments to.
    """
    
    if assignments is None or assignments.shape[0] == 0:
        raise LogicalError, "Method %s: No assignments provided." %(stack()[0][3])
    
    clusters = {}
    for assignment in assignments:
        if assignment not in clusters:
            clusters[assignment] = 1
            
    return clusters.keys()

def stripLastColumn(array):
    """
    Strip the last column from a 2D numpy nd array. Requires that we iterate over 
    the row arrays and strip the last cell from each and every one of them.
    @param array a 2D numpy array
    @return the array stripped of the last column.
    @raise LogicalError if the input array doesn't have at least one column. 
    """
    
    if array.shape[1] < 1:
        raise LogicalError, "Method %s: No columns to strip from array!" %(stack()[0][3])
    
    return np.array([row[:-1] for row in array])

def extractLabelInfo(data):
    """
    @param data a 2D ndarray which represents our data.
    @raise DatasetError if the data provided is None or empty.
    @raise LogicalError if the categoryMeans array computed is of a wrong size.
    @return a list which contains the labels found in the data.
    @return a D-dimensional mean value of all examples of one particular label.
    @return a hash table which maps every example index to its true label.
    """
    
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        raise DatasetError, "Method %s: Caller provided no data." %(stack()[0][3])
    
    exampleHash = {}
    labelDataHash = {}      # will map each label to a 2D ndarray holding the points assigned to it.
    for exIndex in range(data.shape[0]):
        exampleHash[exIndex] = data[exIndex][-1]        # label at the end of the feature vector
        if data[exIndex][-1] not in labelDataHash:
            labelDataHash[data[exIndex][-1]] = np.array([data[exIndex][:-1]]) # Notice how this is initialized to a 2D numpy array
        else:
            labelDataHash[data[exIndex][-1]] = np.append(labelDataHash[data[exIndex][-1]], data[exIndex][:-1])
    
    categoryMeans = [np.mean(labelDataHash[cat], axis = 0) for cat in labelDataHash.keys()]
    if len(categoryMeans) != len(labelDataHash.keys()):
        raise LogicalError, "Method %s: categoryMeans array was not computed properly." %(stack()[0][3])
    
    return labelDataHash.keys(), categoryMeans, exampleHash
        
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
    
    To evaluate the clustering algorithm's accuracy, we will follow two base approaches. We first
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
    @param assignments: N-sized ndarray, mapping each example to its cluter. assignments[i] = k means that
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
    
    if len(trueLabelMeans) != histSize:
        raise LogicalError, "Method %s: trueLabelMeans array should have as many rows as true labels." %(stack()[0][3])
    
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
    
    errorRate = misclassifiedPoints/float(data.shape[0])
    
    # Implement approach #2: Compute true label count histograms and gauge which clusters are "good". 
    # "Good" clusters are closest to the mean of the majority 
    # vote label voted by their points, as reported by the respective histogram.
    
    goodCentroids = 0
    for i in range(len(centroids)):
        #print "Examining cluster %d." %(i + 1)
        # Get the indices of all the points in the cluster
        pointsInCluster = [j for j in range(len(assignments)) if assignments[j] == i]
        if len(pointsInCluster) > 0:
            _clusterHist, majVoteLabel = getClusterHistogram(pointsInCluster, trueLabelHash, histSize)
            #print "Retrieved histogram and majority vote label of cluster %d." %(i+1)
            if closestLabel[i] != None and majVoteLabel == closestLabel[i]:
                goodCentroids+=1 
    
    # Return both metrics to caller.
    
    return errorRate, goodCentroids

def getRandomSample(data, datafraction):
    
    """
    Return a random sample of the provided data, with numExamples examples. This 
    sample is such that every class is represented in the 
    @param data: a 2D ndarray of our data.
    @param datafraction: a float representing the fraction of examples to include from every class.
    @raise DatasetError if data is None or empty
    @raise LogicalError if datafraction is an invalid fraction.
    @return an ndarray with numExamples examples, randomly sub-sampled from data.
    """
    
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        raise DatasetError, "Method %s: No data provided." %(stack()[0][3])
    
    if datafraction < 0 or datafraction > 1:
        raise LogicalError, "Method %s: Fraction of examples provided is invalid (gave: %s)." %(stack()[0][3], str(datafraction))
    
    np.random.permutation(data)
    return data[:np.floor(datafraction * len(data))]
   
    
    
if __name__ == '__main__':
    
    try:
        os.chdir("../")
        np.random.seed()        # uses system time as default
            
        ##### Part 1: Read the Caltech101 data and store it in memory.#####
            
#        DIR_categories=os.listdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000');      
#        imFeatures=[]                                                                             
#        i=0;                                                                                                           
#        for cat in DIR_categories:   
#            #print "Considering category: " + str(cat) + "."                                       
#            if os.path.isdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/' + cat):
#                i = i + 1
#                for gr_im in os.listdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/'+ cat + '/'):
#                    # take grayscale images only
#                    if 'grey.dense' in gr_im:
#                        matfile = loadmat('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/'+ cat + '/' + gr_im)
#                        FV, _binedges = np.histogram(matfile['h'], range(1001))
#                        # append the class label to the feature vector because you will need it later
#                        FV = np.append(FV, i)
#                        imFeatures.append(FV)
#                        
#        # Turn imFeatures into a 2D ndarray so that it can be used in K-means later on
#        imFeatures = np.array(imFeatures)
#        
#        print "Read Caltech data in memory. Now we will store the read data in disk."
#        
        # Dump the data to disk.
        
#        fp = open('proc_data/sift_data_parsed.pkl', 'wb')
#        pkl.dump(imFeatures, fp)
#        fp.close()
        
        
        # Because the SIFT data available to us is more extensive than the gradient data (about 200% of it),
        # we need to sub-sample it so that the results are comparable to the gradient data's. To do this,
        # we will perform a couple of iterations during which we will sub-sample the data at 50% of it
        # and run the same K-means procedure as with the gradient data. We will begin with 10 iterations
        # and we will reduce the number of K-means iterations to 100, to quickly get some results.
        
        fp = open('proc_data/sift_data_parsed.pkl', 'rb')
        SIFTData = pkl.load(fp)
        fp.close()
        
        errorRateList = []
        goodClusterCountList = []
        for iteration in range(10):
            print "Iteration #%d." %(iteration + 1)
            currentExamples = np.array(getRandomSample(SIFTData, 0.5)) # take about half the amount of current examples
            # print "CurrentExamples = " + str(currentExamples)
            # We need to extract the class means as well as the example mappings 
            # from the "currentExamples" ndarray.
            sampleCategories, categoryMeans, exampleHash = extractLabelInfo(currentExamples)
            currentExamples = stripLastColumn(currentExamples)
            # Run K-means
            
            whiten(currentExamples)
            print "Running K-means..."
            codebook, _distortion = kmeans(currentExamples, len(sampleCategories), 10)
            assignments, _distortion = vq(currentExamples, codebook)
            print "Ran K-means"
            if(len(assignments) != currentExamples.shape[0]):
                raise LogicalError, "Method %s: K-means should have computed %d assignments; instead, it computed %d." %(stack()[0][3], currentExamples.shape[0], len(assignments))
            errorRate, goodClusters = evaluateClustering(codebook, currentExamples, assignments, categoryMeans, exampleHash, len(sampleCategories))
            errorRateList.append(errorRate)
            goodClusterCountList.append(goodClusters)
            print "K-means produced an error rate of %.4f%% in iteration %d while computing %d \"good\" clusters.." %(100*errorRate, iteration + 1, goodClusters)
        
        
        print "On average, we had %.4f \"good\" clusters and %.4f%% error rate." %(np.mean(goodClusterCountList), np.mean(errorRateList))
        fp = open('output_data/errorRate_SIFTFeatures.txt', 'w')
        fp.write(str(np.mean(errorRateList)))
        fp.close()
        fp = open('output_data/accurateClusters_SIFTFeatures.txt', 'w')
        fp.write(str(np.mean(goodClusterCountList)))
        fp.close()
        print "That would be all. Exiting..."
        quit()
    except DatasetError as d:
        print "A dataset-related error occurred: " + str(d)
        quit()
    except LogicalError as l:
        print "A logical error occurred: " + str(l)
        quit()
    except Exception as e:
        print "An exception occurred: " + str(e)
        quit()