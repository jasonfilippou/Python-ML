from __future__ import division
'''
Created on Nov 30, 2012

@author: jason
'''

import numpy as np
import os
from inspect import stack
from scipy.cluster.vq import *
from numpy.linalg import norm
from util.mlExceptions import *
from collections import Counter
import pickle as pkl
from evaluation import evaluateClustering

if __name__ == '__main__':
    
    try:
        os.chdir("../")
        np.random.seed()        # uses system time as default
        
        ##### Part 1: Read the Caltech101 data and store it in memory.#####
        
        DIR_categories=os.listdir('input_data/caltech_training_gradients/');       # list and store all categories (classes)
        imFeatures=[]                                                       # this list will store all training examples 
        imLabels=[]                                                         # this list will store all labels
        i=0;                                                                # this integer will effectively be the class label in our code (i = 1 to 101)
        labelNames = []                                           
        categoryMeans = []                                                  # We need the means of all categories to compare them with the cluster means later on
        exampleHash = dict()                                                # We need to map every example to its respective category in O(1) for later operations.
        overAllExampleCounter = 0
        for cat in DIR_categories:                                          # loop through all categories
            if os.path.isdir('input_data/caltech_training_gradients/'+ cat):   
                labelNames.append(cat)
                i=i+1;                                             # i = current class label
                localList = []                                      # will hold only examples of this class (useful for computing means after for loop)
                DIR_image=os.listdir('input_data/caltech_training_gradients/'+ cat +'/');      # store all images of category "cat" 
                for im in DIR_image:                                           # loop through all images of the current category
                    if (not '._image_' in im):                                 # protect ourselves against those pesky Mac OS X - generated files
                        F = np.genfromtxt('input_data/caltech_training_gradients/'+cat+'/'+im, delimiter=' '); # F is now an 2-D numpy ndarray holding all features of an image
                        F = np.reshape(F,21*28);                               # F is now a 588 - sized 1-D ndarray holding all features of the image
                        F = F.tolist();                                        # listify the vector
                        imFeatures.append(F);                                  # store the vector
                        imLabels.append(i);                                    # store the label
                        localList.append(F)
                        exampleHash[overAllExampleCounter] = i                                      # associate example with class in hash.           
                        overAllExampleCounter+=1                        
                        
                # compute and store the category mean
                categoryMeans.append(np.mean(localList, axis = 0))
                try:
                    len(categoryMeans)
                except TypeError:
                    raise LogicalError, "Method %s: categoryMeans should be an iterable." %(stack()[0][3])
                
        # transform the data into a 2-D numpy ndarray to use in kmeans.
        imFeatures = np.array(imFeatures)
        print "Read Caltech data in memory."
        
        # How about we also store the processed data on disk?
        
        fp1 = open('proc_data/gradient_features_traindat.pkl', 'wb')
        fp2 = open('proc_data/gradient_features_trainlabs.pkl', 'wb')
        fp3 = open('proc_data/gradient_features_categorymeans.pkl', 'wb')
        fp4 = open('proc_data/gradient_features_examplehash.pkl', 'wb')
        fp5 = open('output_data/gradient_features_covMat.txt', 'w')     # the covariance matrix should be in our output
        pkl.dump(imFeatures, fp1)
        pkl.dump(imLabels, fp2)
        pkl.dump(categoryMeans, fp3)
        pkl.dump(exampleHash, fp4)
        fp5.write(np.cov(imFeatures))
        fp5.close()
        fp4.close()
        fp3.close()
        fp2.close()
        fp1.close()
        
        ####### Part 2: Run K-means with K = 101 on data and store results on disk. #############
        
        
        # Normalize the features to have variance 1 (k-means requirement).
        
        whiten(imFeatures)
        
        # Run k-means 500 times (the default) on the data with aim to produce k = 101 clusters.
        # The stopping criterion of each iteration is a difference in the computed distortion
        # (mean squared error) less than e-05 (the default)
        
        print "Running K-means..."
        codebook, _distortion = kmeans(imFeatures, 101, 100)
        assignments, _distortion = vq(imFeatures, codebook)
        pkl.dump(codebook, open('proc_data/codebook_kmeans_gradients.pkl', 'wb'))
        pkl.dump(assignments, open('proc_data/labelAssignments_kmeans_gradients.pkl', 'wb'))
        if(len(assignments) != imFeatures.shape[0]):
            raise LogicalError, "Method %s: K-means should have computed %d assignments; instead, it computed %d." %(stack()[0][3], imFeatures.shape[0], len(assignments))
        print "Ran K-means"
        errorRate, goodClusters, avgEntropy = evaluateClustering(codebook, imFeatures, assignments, categoryMeans, exampleHash, 101)
     
        print "K-means produced an error rate of %.4f%%, %d \"good\" clusters and %.4f average entropy." %(errorRate, goodClusters, avgEntropy)
        print "The amount of \'\"good\" clusters corresponds to %.4f%% of total clusters." %(100*goodClusters/float(101))
        
        fp = open('output_data/errorRate_gradients_kmeans.txt', 'w')
        fp.write(str(errorRate))
        fp.close()
        fp = open('output_data/accurateClusters_gradients_kmeans.txt', 'w')
        fp.write(str(goodClusters))
        fp.close()
        fp = open('output_data/averageEntropy_gradients_kmeans.txt', 'w')
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