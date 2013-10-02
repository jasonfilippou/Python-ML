'''
Created on Dec 5, 2012

@author: jason
'''

import pickle as pkl
import numpy as np
import os
from util.mlExceptions import *
from inspect import stack
from collections import Counter
from numpy.linalg import norm
from sklearn import mixture
from scipy.io import loadmat
from evaluation import evaluateClustering


def get_codebook(predictedLabs):
    '''
    Get a codebook from a mapping of assignments. The codebook effectively
    consists of the discrete values of the input array.
    @param predictedLabs a N x D numpy array which holds the predicted labels for every example
        of our test data.
    '''
    
    uniqueVals = dict()
    for val in predictedLabs:
        if val not in uniqueVals:
            uniqueVals[val] = 1
    
    return uniqueVals.keys()

if __name__ == '__main__':
    try:
        os.chdir('../')
        
        ##### Part 1: Read the Caltech101 data and store it in memory.#####
            
        DIR_categories=os.listdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000');      
        imFeatures=[]                                                                             
        i=0;               
        exampleHash = dict()       
        exampleIndex = 0   
        categoryMeans = []                                                                                  
        for cat in DIR_categories:   
            #print "Considering category: " + str(cat) + "."                                       
            if os.path.isdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/' + cat):
                i = i + 1
                localList = []
                for gr_im in os.listdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/'+ cat + '/'):
                    # take grayscale images only
                    if 'grey.dense' in gr_im:
                        matfile = loadmat('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/'+ cat + '/' + gr_im)
                        FV, _binedges = np.histogram(matfile['h'], range(1001))
                        imFeatures.append(FV)
                        localList.append(FV)
                        exampleHash[exampleIndex] = i
                        exampleIndex +=1
                        
                # Compute and store the category means
                categoryMeans.append(np.mean(localList, axis = 0))    
                   
        # Turn imFeatures into a 2D ndarray so that it can be used in K-means later on
        imFeatures = np.array(imFeatures)
        print "Read Caltech data in memory."
    
        g1 = mixture.GMM(n_components=101,thresh = 1e-05, covariance_type='diag') 
        print "About to fit data"
        g1.fit(imFeatures)
        pkl.dump(g1, open('proc_data/gmm_obj_diag_cov_sift.pkl', 'wb'))
        print "Fitted data"
        predLabels= g1.predict(imFeatures)
        print "Predicted data"
        predMeans = g1.means_
        
        errRate, goodClusters, avgEntropy = evaluateClustering(g1.means_, imFeatures, predLabels, categoryMeans, exampleHash, 101)

        print "GMM model predicted labels with an error rate of %.4f%%, produced %d \"accurate\" clusters and %.4f average entropy." %(errRate, goodClusters, avgEntropy)
        print "That's all. Exiting..."
        quit()
    except Exception as exc:
        print "An exception occurred:"  + str(exc) + "."
        quit()