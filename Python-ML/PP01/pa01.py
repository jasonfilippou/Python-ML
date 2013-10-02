import numpy as np
import pickle as pkl

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
        Classify a set of samples

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
        xy = self.traindat.dot(testdat.T)

        # compute norms
        xx = np.sum(self.traindat * self.traindat, 1)
        yy = np.sum(testdat * testdat, 1)

        # now iterate over testing samples
        out = np.empty((ntest_samples, len(k)))
        for i in range(ntest_samples):
            # compute distance to all training samples
            dists = np.sqrt(xx - 2*xy[:,i] + yy[i])

            # find the indexes that sort the distances
            sorted_indexes = np.argsort(dists)

            # now iterate over k-values to compute labels
            thesum = 0
            start = 0
            for j in range(len(k)):
                cur_k = k[j]

                # add votes up to the current k value
                for l in range(start, cur_k):
                    thesum = thesum + self.trainlabs[sorted_indexes[l]]

                # tally the votes
                out[i,j] = np.sign(thesum)
                start = cur_k

        # massage the output if only one k was used
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

def load(file):
    """
    Load k-nn classifier object from file

    """
    try:
        fp = open(file,'rb')
    except IOError as e:
        'Pickling knn failed error({0}): {1}'.format(e.errno, e.strerror)
        return None

    try:
        obj = pkl.load(fp)
    except AttributeError as e:
        print ('Pickling knn failed for file ' + file + ' ' + str(e))
        fp.close()
        return None
    except pkl.UnpicklingError as e:
        print str(e)
        fp.close()
        return None

    fp.close()
    return obj