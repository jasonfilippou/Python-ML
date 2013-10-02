'''
Created on Sep 23, 2012

@author: Jason
'''

from knn import KNN
import numpy as np

''
class WDKNN(KNN):
    '''
    WDKNN, which stands for "Weighted Distance K-Nearest Neighbors", is a subclass of KNN, which 
    simply stands for "K-Nearest Neighbors".
    
    The only method of the superclass that we will override is the "classify" method, so that 
    the n-th neighbor (where n = 1, ... , K) gets a vote that is an exponential function of 
    its distance from the test point 
    '''
    
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


    
    
        