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
from evaluation import evaluateClustering   
    
    
if __name__ == '__main__':
    
    try:
        os.chdir("../")
        np.random.seed()        # uses system time as default
            
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
        
        print "Read SIFT data in memory. Now we will store the read data on disk."
        
        #Dump the data to disk.
        
        fp1 = open('proc_data/sift_data_parsed.pkl', 'wb')
        fp2 = open('proc_data/sift_data_exampleHash.pkl', 'wb')
        fp3 = open('proc_data/sift_data_categoryMeans.pkl', 'wb')
        fp4 = open('output_data/sift_data_covMat.txt' , 'w') # We want the covariance matrix in our output data.
        pkl.dump(imFeatures, fp1)
        pkl.dump(exampleHash, fp2)
        pkl.dump(categoryMeans, fp3)
        fp4.write(np.cov(imFeatures))
        fp4.close()
        fp3.close()
        fp2.close()
        fp1.close()
        
        # Done with dumping the data. Now, re-read it (the previous part will be
        # uncommented after the first execution).
        
        fp1 = open('proc_data/sift_data_parsed.pkl', 'rb')
        fp2 = open('proc_data/sift_data_exampleHash.pkl', 'rb')
        fp3 = open('proc_data/sift_data_categoryMeans.pkl', 'rb')
        
        SIFTData = pkl.load(fp1)
        exampleHash = pkl.load(fp2)
        categoryMeans = pkl.load(fp3)
        fp1.close()
        fp2.close()
        fp3.close()
        
        # Run K-means
            
        whiten(SIFTData)
        print "Running K-means..."
        codebook, _distortion = kmeans(SIFTData, 101, 100)
        pkl.dump(codebook, open('proc_data/codebook_k-means_SIFT.pkl', 'wb'))
        assignments, _distortion = vq(SIFTData, codebook)
        pkl.dump(assignments, open('proc_data/assignments_k-means_SIFT.pkl', 'wb'))
        print "Ran K-means"
        if(len(assignments) != SIFTData.shape[0]):
            raise LogicalError, "Method %s: K-means should have computed %d assignments; instead, it computed %d." %(stack()[0][3], SIFTData.shape[0], len(assignments))
        errorRate, goodClusters, avgEntropy = evaluateClustering(codebook, SIFTData, assignments, categoryMeans, exampleHash, 101)
        print "K-means produced an error rate of %.4f%%, computed %d \"good\" clusters and introduced an average entropy of %.4f." %(errorRate, goodClusters, avgEntropy)
        fp = open('output_data/errorRate_K-means_SIFTFeatures.txt', 'w')
        fp.write(str(errorRate))
        fp.close()
        fp = open('output_data/accurateClusters_K-means_SIFTFeatures.txt', 'w')
        fp.write(str(goodClusters))
        fp.close()
        fp = open('output_data/averageEntropy_K-means_SIFTFeatures.txt', 'w')
        fp.write(str(avgEntropy))
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