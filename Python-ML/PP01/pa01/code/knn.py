'''
Created on Sep 23, 2012

@author: Jason

The KNN class has been shamelessly copied from the code provided to us by Hector Bravo,
and modifications have taken place.
'''

import numpy as np
import pickle as pkl
from mlExceptions import LogicalError

class KNN:
    
    def __init__(self, traindat, trainlabs, k=5):
        """
        Creates an instance of class KNN. Stores training data

        Arguments:
          traindat: pandas.DataFrame
          trainlabs: pandas.Series (+1/-1, currently unchecked)


        """
        self.features = traindat.columns
        self.traindat = traindat.values
        self.trainlabs = trainlabs.values
        self.k = k

    def __str__(self):
        return ('A %d-nn classifier on features ' % self.k) + str(self.features)


    def classify(self, testdat, k=None):
        
        """
        Description: Classify a set of samples.

        Arguments:
          testdat: pandas.DataFrame
          k: None, integer, or integer list of ascending k values

        Returns:
          matrix of (+1/-1) labels (if k is a list)
          list of labels, if k is integer

        """
        
        testdat = testdat.values
        ntest_samples = testdat.shape[0]

        if k is None:
            k = self.k

        # check if k is an integer, if so wrap into list
        try:
            len(k)
        except TypeError:
            k = [k]

        # compute cross-products of training and testing samples
        # This is done in order to exploit vectorized implementations
        # when computing the distances.
        
        xy = self.traindat.dot(testdat.T)

        # compute norms. Also useful for computing the distance
        xx = np.sum(self.traindat * self.traindat, 1)
        yy = np.sum(testdat * testdat, 1)

        # now iterate over testing samples
        out = np.empty((ntest_samples, len(k)))
        
        for i in range(ntest_samples): # for every testing example
            
            # Compute distance to all training samples
            # note: this is where we use a vectorized implementation
            # instead of for looping
            
            dists = np.sqrt(xx - 2*xy[:,i] + yy[i])

            # Find the indexes that sort the distances. 
            # You need to do this to find the K nearest
            # neighbors, i.e the K neighbors with the
            # smallest distance to the test point.
            
            sorted_indexes = np.argsort(dists)

            # Now iterate over the first k values to compute labels
            
            thesum = 0
            start = 0
            for j in range(len(k)):                                     # index the list of values of K
                cur_k = k[j]                                            # the current value of K is cur_k

                # Add votes up to the current k value. Lines 7 to 10 
                # of Algorithm 3 in CIML chapter 2.
                
                for l in range(start, cur_k):                           # for every neighbor from 1 to K
                    thesum = thesum + self.trainlabs[sorted_indexes[l]] # calculate the contribution of every neighbor

                # Tally the votes.
                
                out[i,j] = np.sign(thesum)
                start = cur_k

        # end of examples loop
        
        # Massage the output if only one k was used
        
        if len(k) == 1:
            out = out.reshape(ntest_samples)

        return out

    def tune(self, tunedat, tunelabs, k=range(1,12,2)):
        """
        Tune a k-nn classifier

        Arguments:
          tunedat: pandas.DataFrame a tuning set
          tunelabs: pandas.Series labels for tuning set
          k: a list of increasing integer k values

        Returns:
          Nothing

        Side effect:
          sets self.k to the value of k that minimizes error on the tuning set
          sets self.tuning_k to the set of k values tested
          sets self.tuning_err to the tuning set error for each of the k values

        """
        tunelabs = tunelabs.values
        ntune_samples = tunedat.shape[0]

        self.tuning_k = k
        self.tuning_err = np.empty(len(self.tuning_k))
        predlabs = self.classify(tunedat, self.tuning_k)


        for i in range(len(k)):
            self.tuning_err[i] = np.mean((tunelabs * predlabs[:,i]) < 0)

        self.k = k[np.argmin(self.tuning_err)]

    def classifyWithAllK(self, testData):
        '''
        Description: Similarly to DecisionTree.classifyWithAllDepths(), use all tuned hyperparameters
                to test your algorithm on one particular piece of testing Data, and store the outcome.
        Arguments: 
            testData: pandas.DataFrame
        Returns: None
        
        '''
        self.testing_k = []                             # A list that will hold the data procured by testing
        testLabs = testData['spam'].values
        if testData.shape[0] == 0:
            raise LogicalError("Cannot classify an empty dataset.")
        if self.tuning_k == None:
            print "CLassifier has not been tuned, using the default value of k for classifying:"
            classifications = self.classify(testData, self.k)
            testingError = np.mean ( (testLabs * classifications) < 0)
            print "Testing error of " + str(self.K)+ "-nearest neighbors is: " + str(testingError) + "."
            self.tests.append((self.k, testingError))
        else:
            tuningKindex = 0
            # self.classify returns a matrix of classifications when the second argument is a list
#            for classification in self.classify(testData, self.tuning_k):
#                print "Classifying with " + str(self.tuning_k[tuningKindex]) + " - nearest neighbors."
#                print "For this value of K, the tuning error was: " + str(self.tuning_err[tuningKindex]) + "."
#                print "classification: " + str(classification)
#                print "testLabs: " + str(testLabs)
#                testingError = np.mean ( (testLabs * classification) < 0)
#                print "Testing error was: " + str(testingError) + "."
                # self.tests.append((self.tuning_k[tuningKindex], testingError))
            for kVal in self.tuning_k:
                print "Classifying with " + str(kVal) + " - nearest neighbors."
                
    def dump(self, file):
        """
        Store k-nn classifier object to file

        """
        try:
            fp = open(file,'wb')
            pkl.dump(self, fp)
            fp.close()
        except Exception as e:
            'Pickling failed for object ' + str(self) + ' on file ' + file + ' Exception: ' + e.message        