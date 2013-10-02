from __future__ import division
'''
Created on Oct 10, 2012

@author: Jason

'''

'''
Honorary mention: Fellow student Gregory Kramida mentioned on the online forum Piazza that, when having to deal with OVA
classification, it is a better idea to do over sampling than sub-sampling to balance the classes. Had it not been for his
input, I would have implemented this class with undersampling, which is not a good way to go, because we make a uniformity
assumption about the data generation distribution of our training data.
'''

from mlExceptions import DatasetError, LogicalError
import numpy as np
import pickle as pk
import copy
import sys

'''
Some methods that will help us change labels rapidly
through the filter() Python tool:
'''

def labelPositive(example):
    example[-1] = 1
    
def labelNegative(example):
    example[-1] = -1
    
class OVA:
    
    '''
        The OVA (One-Versus-All) class contains a list of underlying classifiers. It provides methods for training 
        the K different classifiers required by the OVA scheme, classifying a test point and tuning all classifiers under the hood.
        The tuning and testing algorithms have been sketched in CIML, chapter 5,  page 73. The OVA class is agnostic to the 
        underlying classifiers; the aim is to make it able to be run no matter what binary classifiers it holds. It does make a 
        couple of assumptions on the functionality  provided by the underlying classifiers, though. The following is a list of said 
        assumptions:
    
        (1) The binary classifiers should have a train() method, with a list-of-lists argument representing training data.
        (2) Similarly to (1), the binary classifiers should have a tune() method which accepts both training data as described above,
            as well as another list-of-lists argument, this time representing tuning data.
        (3) The underlying classifiers should be able to classify test points and do so through a classify() method, which takes 
            as its parameter the test point feature vector, represented as a list.
        (4) The underlying classifiers should be able to alter the value of their hyper-parameters, and do so through a setHyperparam()
            method which takes as its argument the desired value of the hyper-parameter. It is the classifier's job to report a type mismatch
            regarding its hyper-parameter to the user, through exceptions.
    '''


    def __init__(self, trainingData, tuningData, classNum):
        
        '''
        Constructor. Training data provided, but not checked until training time (lazy approach).
        Arguments: trainingData: list of lists (feature vectors) representing training data.
                    tuningData: similarly, for tuning data
                    classNum: The number of different classes that we have to discern between.
        Returns: self
        '''
        if trainingData == None or len(trainingData[0]) == 0 :
            raise "Please provide the OVA classifier with some training examples"
        if classNum <= 1:
            raise LogicalError("Please provide the OVA classifier with at least 2 classes.")
        self.classifiers = []                                                                  # list of classifiers, initially empty
        self.trainingData = trainingData
        self.tuningData = tuningData
        self.classNum = classNum
        
    def __str__(self):
        
        '''
        Description: stringifying method. Returns a string representation of the OVA object
        Arguments: none
        Return value: the string representation of self
        '''
        
        return "OVA object containing " + str(self.classNum) + " classifiers and a training dataset with " + str(len(self.trainingData)) + " examples."
  
     
    def addClassifier(self, classifier):
        
        '''
        Description: add a classifier to the classifier pool.
        Arguments: classifier: an object representing a binary classifier, with train(), classify() and tune() methods.
        Return type: none.
        '''
        
        self.classifiers.append(classifier)
         
    def train(self):
        
        '''
        Description: We will implement the training algorithm for OVA classification as sketched in CIML, chapter 5 p.73. We have extended
                the implementation to use oversampling.
        Arguments: labelCardinalities: a list containing the cardinalities of every class in the training data. Needed to perform 
                oversampling.
        Return value: None, but significant side-effects: all classifiers trained, considering class i (i = 1, ..., K) as positive and all other
            classes j (j = 1, ...., i -1, i + 1, .... K) as negative. 
        '''
        
        if len(self.trainingData) == 0:
            raise DatasetError("Please provide the OVA classification algorithm with some training examples.")
        
        if len(self.trainingData[0]) == 0:
            raise DatasetError("Please provide non-empty feature vectors")                          
        
        # The following for loop implements the OVA classification scheme, enhanced with the idea of over-sampling.
        
        for label in range(1, self.classNum + 1):
            
            # Retrieve all the examples of the current class and all others.
            
            examplesOfClass = copy.deepcopy([examples for examples in self.trainingData if examples[-1] == label])         
            examplesOfOtherClasses = copy.deepcopy([examples for examples in self.trainingData if examples[-1] != label])
           
            # Use the filter() tool to quickly change the labels of the datasets.
            
            filter(labelPositive, examplesOfClass)
            filter(labelNegative, examplesOfOtherClasses)
             
            # To implement oversampling, we need to compute the difference between the 
            # cardinalities of the positive and negative classes and then repeat the minority
            # class to cover up this difference. We will shuffle all examples in the end to 
            # avoid introducing a biased example order to the underlying classifiers' training 
            # algorithms.
            
            cardinalityDifference = abs(len(examplesOfOtherClasses) - len(examplesOfClass))
            smallerClass = examplesOfClass if min([len(examplesOfClass), len(examplesOfOtherClasses)]) == len(examplesOfClass) else examplesOfOtherClasses  
            largerClass = examplesOfClass if max([len(examplesOfClass), len(examplesOfOtherClasses)]) == len(examplesOfClass) else examplesOfOtherClasses
            
    
            # Repeat the smaller class floor(cardinalityDifference/count_of_class) times
            # and add the larger class afterwards.
                   
            overSampledData = np.floor(cardinalityDifference / len(smallerClass)) * smallerClass +  largerClass
            
            # Shuffle the data before handing it to the binary classifier,
            # since it's not guaranteed that it will be shuffling it itself
            # (like the perceptron is doing).
            
            np.random.shuffle(overSampledData)
            
            # Train with the over-sampled data
            
        
            self.classifiers[label - 1].train(overSampledData) 
 
            
    def classify(self, testPoint):
        
        '''
        Description: Classifies a test point. As sketched on CIML, chapter 5, page 73.
        Arguments: testPoint: a list representing the feature vector corresponding to the test point to be classified.
        returns: the class label of testPoint. 
        '''
        
        # We will run the test point through all classifiers and 
        # predict its class based on the classifier that produced the highest score.
        # In order to avoid ties, it is preferred that the underlying binary classifier
        # returns a confidence instead of a  0 - 1 response, but np.argmax() will take care
        # of this either way.
        
        score = np.zeros(self.classNum).tolist()
        for classLabel in range(self.classNum):                                
            score[classLabel] =  score[classLabel]+ self.classifiers[classLabel].classify(testPoint)       
        
        return np.argmax(score) + 1                                                     # careful with zero - indexing!
    
    def test(self, testData):
        
        '''
        Description: Testing method. We classify all examples in testData and measure testing error.
        Arguments: testData: a list of lists representing a testing set.
        Returns: the testing error computed.
        '''
        
        if testData == None or len(testData) == 0:
            raise DatasetError("Cannot test on empty or null test data.")
        if len(testData[0]) != len(self.trainingData[0]) :                  # feature vector mismatch between training and testing data
            raise DatasetError("The testing data format has to match the training data format.")
        
        misClassifications = 0
        for testPoint in testData:
            if self.classify(testPoint[:-1]) != testPoint[-1]:             # misclassification
                misClassifications = misClassifications + 1
        return misClassifications / len(testData)                       # error rate
    
    
    def tune(self):
        
        '''
        Description: Tuning method. We run the tuning methods of all classifiers by using tuningData.
            We will once again resort to oversampling to train with balanced datasets.
            positive class by a necessary amount in order to make the two classes balanced, and feed this artificially balanced
            dataset to the underlying classifier's tuning method.
            
        Arguments: tuningData: a 2D array (list of lists) that represents tuning data.
                                         
        Return value: None, but there exists the side effect of having all classifiers tuned.
        '''                         
        
        if self.tuningData ==  None or len(self.tuningData[0]) == 0:
            raise DatasetError("Please provide the OVA classification algorithm with some tuning examples.")                          
        
        for label in range(1, self.classNum + 1):                   # 1 to 101
            
            ###############################
            # Part 1: Set up training data
            ###############################
            
            # This part is exactly the same as the procedure in self.train()
            examplesOfClass = copy.deepcopy([examples for examples in self.trainingData if examples[-1] == label])         
            examplesOfOtherClasses = copy.deepcopy([examples for examples in self.trainingData if examples[-1] != label])

            filter(labelPositive, examplesOfClass)
            filter(labelNegative, examplesOfOtherClasses)
            
            cardinalityDifference = abs(len(examplesOfOtherClasses) - len(examplesOfClass))
            smallerClass = examplesOfClass if min([len(examplesOfClass), len(examplesOfOtherClasses)]) == len(examplesOfClass) else examplesOfOtherClasses  
            largerClass = examplesOfClass if max([len(examplesOfClass), len(examplesOfOtherClasses)]) == len(examplesOfClass) else examplesOfOtherClasses
            overSampledTrainingData = np.floor(cardinalityDifference / len(smallerClass)) * smallerClass +  largerClass
            np.random.shuffle(overSampledTrainingData)
            
            ############################
            # Part 2: Set up tuning data
            ############################
            
            # We need to pull the tuning examples of the current class, label them positive,
            # then pull the examples of all other classes, label them negative. 
            
            examplesOfClass = copy.deepcopy([examples for examples in self.tuningData if examples[-1] == label])         
            examplesOfOtherClasses = copy.deepcopy([examples for examples in self.tuningData if examples[-1] != label])            
            filter(labelPositive, examplesOfClass)
            filter(labelNegative, examplesOfOtherClasses)
            
            # Combine positive and negative examples into one dataset,
            # shuffle it (TODO: do I need to shuffle here?)
            # and run the classifier's tuning method with both the training
            # and tuning data.
            
            binarizedTuneData = examplesOfClass + examplesOfOtherClasses
            np.random.shuffle(binarizedTuneData)        
            self.classifiers[label -1].tune(overSampledTrainingData, binarizedTuneData)    
            print "Tuned classifier for class: " + str(label) + "."
            
    def setAllHyperparams(self, value):
        
        """
        Description: Set the hyperparameters of all classifiers to value.
                    In this case, because we're running out of time, we will just use the information about the Perceptron's maxIter hyperparameter
                    even though it's "private" to the classifier.
        Parameters: value: int, the hyperparameter value
        Returns: None
        """
        for classifier in self.classifiers:
            classifier.setHyperParam(value)
            
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
            
    def printInfo(self):
        
        '''
        This method is only useful for debugging: We take a peek at what's inside the OVA object to make
        sure we're not making any invalid assumptions
        '''
        print "Printing data about: " + str(self)
        
        print "We have stored a total of: " + str(self.classNum) + " classes."
        for label in range(1, self.classNum + 1):
            examplesOfLabel = [examples for examples in self.trainingData if examples[-1] == label]
            print "There are: " + str(len(examplesOfLabel)) + " examples of class " + str(label)
         
        print "That's all the data that we have stored"