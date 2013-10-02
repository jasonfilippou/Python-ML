'''
Created on Oct 7, 2012

@author: Jason
'''

from __future__ import division
import numpy as np
from mlExceptions import LogicalError, DatasetError
import pickle as pk

'''
A vector normalization utility that we will use at testing time
'''

def normalize(vector):
    return vector / np.linalg.norm(vector)

class AveragedPerceptron:
    
    '''
    This class represents the "averaged" perceptron classifier, as discussed
    on CIML. It provides methods to train, test and tune its hyperparameters.
    
    A perceptron needs:
    
        training data to train on
        tuning data to tune on
        testing data to evaluate on
        a hyperparameter MaxIter, which controls overfitting and underfitting
        
    The averaged perceptron is a bit different from the classic perceptron in terms of
    the testing and training methods. We will implement algorithm 7 from CIML page 48. 
    '''
    
    def __init__(self,  maxIterList = range(1, 101)):
        
        '''
        Description: Constructor. Note that we will do sanity checking lazily: i.e we will not 
                check whether the training data provided is non-null here, but rather until 
                training time comes (if it ever comes).
        Arguments:
            maxIterList (default [1, 2,...,100]: list of all possible hyper-parameter values to select from during tuning. 
            
        Return value: self
        
        '''
        
        # Check if the user provided us with a hyper-parameter list to tune on. If he supplied an integer instead, wrap the 
        # integer value into a list (this trick was shown by Hector in his KNN pseudocode in the first programming project).
        
        try:
            len(maxIterList)
        except TypeError:
            maxIterList = [maxIterList]                 
        
        # We are now sure that we have a list of hyper-parameters to find the optimal one, even if it is a single-element list
        
        self.hyperParameterList = maxIterList
        
        # Set the default hyper-parameter to -1 to have a flag that indicates that the perceptron hasn't been trained yet.
        
        self.maxIter = -1
         
    def __str__(self):
        
        '''
        Description: stringifying method. Returns a string representation of the current classifier
        Arguments: none
        Return value: the string representation of self
        '''
        return "An averaged perceptron classifier with maxIter = " + self.maxIter
        
    def train(self, trainingData = None):  
        
        '''
        Description: Train the averaged perceptron classifier on the training data provided. We will
            use the algorithm of CIML, which we will enhance with example shuffling.
        Arguments: trainingData , default none. Never used, because we give this data through the constructor.
                Only reason for which we define this parameter is because the overlaying 
        Return value: The averaged weight vector and bias (also stored within the class itself)
        
        '''
 
        # if the training Data provided is null or empty,
        # then the stored training data had better be non-null and non-empty.
        if trainingData == None or len(trainingData) == 0:
            if self.trainingData == None or len(self.trainingData) == 0:
                raise DatasetError("Cannot train on a null or empty dataset.")
        
        self.trainingData = trainingData if trainingData != None and len(trainingData) > 0 else self.trainingData
        
        if self.maxIter == -1:                                                      # maxIter = -1 means that we haven't tuned 
            #print "The current perceptron hasn't been tuned to find an optimal hyper-parameter."
            #print "We will train it using a default value of 5 for the MaxIter hyper-parameter."
            self.maxIter = 5
        
        dimensions = len(self.trainingData[0]) - 1                                  # subtracting 1 for the label
        self.w = np.zeros(dimensions).tolist()                                      # w = <0, 0, ... 0> 
        self.b = 0                                                                  # b = 0
        u = np.zeros(dimensions).tolist()                                           # cached weight vector
        beta = 0                                                                    # cached bias
        counter = 1
        for _ in range(self.maxIter):
            np.random.shuffle(self.trainingData)                                    # the perceptron benefits from data reshuffling at every iteration
            for x in self.trainingData:                                             # assuming that the label is contained within training data
                y = x[-1]                                                           # fetch the true label y
                a = np.dot(self.w, x[:-1]) + self.b                                 # compute activation for current example
                if y * a  <= 0:
                    self.w = self.w + y * np.array(x[:-1])                          # turn the second summand into an ndarray to achieve desired effect
                    self.b = self.b + y
                    u = u + y * counter * np.array(x[:-1])                          # similarly to above
                    beta = beta + y * counter
                counter = counter + 1
              
        # We will both store and return the averaged weight vector and bias.
        # We will leave it unnormalized, but make sure that the normalized
        # version is used when classifying.
        
        self.w = (self.w - (1/counter) * u).tolist()
        self.b = self.b - (1/counter) * beta
        
        #print "Perceptron trained!"
        return (self.w, self.b)
        
        
    def classify(self, xtest):
        
        '''
        Description: Classify the test point xtest.
        Arguments: xtest: a feature vector (list of feature values) that represents the test point.
        Return value: The signed distance of the testPoint to the hyperplane. We choose to return the entire
                    distance itself instead of the sign of the distance because we will use the
                    Averaged Perceptron as part of an overlaying OVA classification scheme.
                    So, if we were to simply return the sign, we would have multiple ties to deal
                    with (see testing algorithm in chapter 5, page 73 of CIML). If, instead, we return
                    the signed distance, we have a measure of how "close" to the relevant hyperplane the
                    test point is, which helps avoid ties.
        '''
        if self.maxIter == -1:                                                                  # no training has taken place
            raise LogicalError("Perceptron not trained.")
        return np.dot(normalize(self.w), xtest) + (self.b / np.linalg.norm(self.w))             # normalizing the parameters
        
        
    def test(self, testingData):
        
        '''
        Description: Estimate the performance of the Averaged Perceptron on testing data.
        Arguments: testingData: a list of lists representing data to test on.
        Return value: the testing error on this data (float) 
        '''
        if self.maxIter == -1:                                                                  # no training has taken place
            raise LogicalError("Perceptron not trained.")
        errors = 0
        for example in testingData:
            label = example[-1]
            if label != np.sign(self.classify(example[:-1])):                                   # give feature vector to classification method
                errors += 1
                
        return errors / len(testingData)                                                        # error rate
    
    def tune(self, trainingData, tuningData):
          
        '''
         Description: 
                     Tune the AveragedPerceptron classifier on the tuningData provided. The idea of tuning consists of training 
                 different classifiers, each one with a different hyper-parameter value, on the same training data (already present in our class),
                 then estimating the error on tuningData, and selecting the (classifier, hyper-parameter) pair for which error on the
                 tuningData was the smallest.
                 
                     Similarly to project 1, we will keep a store of all classifiers trained (in this case, each classifier consists of a (weight, bias)
                pair) so that we can test their performance on the testing data. We do this for statistical reasons, because we can never be 100% sure
                that the tuning-optimal classifier will also be the testing-optimal classifier.                
                
         Arguments:  trainingData, A set of feature vectors, represented by a list of lists, which represents the data on which we train.
                     tuningData: A set of feature vectors, represented by a list of lists, which represents the data on which we elect to tune
                    our classifier.
                    
         Return value: None. There does exist the "side-effect" of setting the hyper-parameter MaxIter, though, as well as storing the classifiers trained. 
        '''
        
        self.trainedClassifiers = []                                                        # a list of (w, b, MaxIter, tuningError) tuples
        self.trainingData = trainingData
        self.tuningData = tuningData
        
        for maxIterConsidered in self.hyperParameterList:
            
            # store current classifier trained
            
            self.maxIter = maxIterConsidered                                        # this is an implementation choice: our train() method needs self.maxIter
            (currentw, currentb) = self.train()
            currentErr = self.test(tuningData)
            self.trainedClassifiers.append((currentw, currentb, maxIterConsidered, currentErr))
            print "Trained an averaged perceptron classifier with hyperparameter MaxIter = " + str(maxIterConsidered) + "."
            print "Tuning error was: " + str(currentErr) + "."
            
            
        # sort the classifiers according to error rate, with best ones first 
        # store the best classifier's data as the default data of the current
        # object
        
        self.trainedClassifiers = sorted(self.trainedClassifiers, key = lambda data: data[3])
        self.w = self.trainedClassifiers[0][0]
        self.b =  self.trainedClassifiers[0][1]       
        self.maxIter = self.trainedClassifiers[0][2]
        self.lowestTuningError = self.trainedClassifiers[0][3]
        
        
    def setHyperParam(self, value):
        
        """
        Description: Set the value of the maxIter hyper-parameter to "value".
        Arguments: value, an integer hyper-parameter
        Returns: None
        """
        
        if not isinstance(value, int):
            raise LogicalError, "The Perceptron's maxIter hyper-parameter is an integer value."
        self.maxIter = value
        
    def dump(self, filePath):
        
        """
        Description: store the current object to a file. Stolen verbatim from project 1 description. Pickle library used.
        Arguments:
            filePath: String representing the path to the output file.
        Return value: None
        """
        try:
            fp = open(filePath,'wb')
            pk.dump(self, fp)
            fp.close()
        except Exception as e:
            'Pickling failed for object ' + str(self) + ' on file ' + file + ' Exception: ' + e.message
