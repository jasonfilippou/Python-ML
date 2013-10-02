'''
Created on Dec 5, 2012

@author: jason
'''

import pickle as pkl
import numpy as np
import os
from util.mlExceptions import *
from inspect import stack
from gmm_diag2_new import GMM
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
        g1 = GMM(n_components=101, thresh = 1e-05, covariance_type='full', n_iter=10) 
        print "About to fit data"
        g1.fit(imFeatures)
        pkl.dump(g1, open('proc_data/gmm_obj_diag2_cov_gradients_new.pkl', 'wb'))
        print "Fitted data"
        #g1 = pkl.load(open('proc_data/gmm_obj_diag2_cov_sift.pkl', 'rb'))
        predLabels= g1.predict(imFeatures)
        print "Predicted data"
        predMeans = g1.means_
        errRate, goodClusters, avgEntropy = evaluateClustering(predMeans, imFeatures, predLabels, categoryMeans, exampleHash, 101)

        print "GMM model predicted labels with an error rate of %.4f%%, produced %d \"accurate\" clusters and %.4f average entropy." %(errRate, goodClusters, avgEntropy)
        print "That's all. Exiting..."
        quit()
    except Exception as exc:
        print "An exception occurred:"  + str(exc) + "."
        quit()