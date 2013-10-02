'''
Created on Sep 17, 2012

@author: jason
'''

from __future__ import division             # used to receive accurate floating-point quotients when dividing integers/longs
import numpy as np
import pickle as pk
import pandas as pd
import copy
import util
import gc
import mlExceptions as mlex
from mlExceptions import LogicalError

class Node:
    
    '''
    This class represents both the decision tree's inner and leaf nodes.
    Leaf nodes are discerned by the "isLeaf" boolean variable and are also
    labeled with the prediction ("label") variable for that particular path
    of the decision tree. They are connected to their children by Node pointers
    "left" and "right", which are null by default and remain such for leaf nodes.
    
    '''
    left, right = None, None
    
    def __init__(self, feature, split_point, left = None, 
                 right = None, isLeaf = False, label = None ):
        '''
         Constructor
        '''
        self.feature = feature                           # feature splitted at (only makes sense for inner nodes)
        self.split_point = split_point                   # split point for feature (only makes sense for inner nodes)
        self.isLeaf = isLeaf                             # boolean flag to indicate a leaf node
        self.label = label                               # label of example (only makes sense for leaf nodes)
        self.left = left                                 # pointers to left and right children (null for leaf nodes)
        self.right = right
        
class DecisionTree: 
    
    '''
    The DecisionTree class is the implementation of the Decision Tree classifier. Included
    are private methods for computation of the optimal feature at each step, creation of inner
    and leaf nodes, as well as public methods for training, tuning and testing the classifier.
    
    The decision tree represented by this class is a shallow decision tree of depth "depth" and 
    has been extended to use both score and information gain to estimate the split point quality.
    
    The "root" Node object represents the root of the tree. It is initialized to None by the
    constructor of the class.
    
    The two most interesting methods in this class are:
        1) __sortFeatures__ : Sorting of the dataset's features according to score or information gained. This
            method implements the algorithm described on "PA01.ipynb".
        2) __trainRec__ : Implementation of the decision tree training algorithm, as described on CIML.
        
    '''
    def __init__(self, dataset, depth = 5, useInfoGain = False):
        
        '''
        Description: Constructor.
        
        The constructor only checks to see whether the depth parameter is above zero. We follow a lazy approach with
        respect to the dataset being empty or not, by making this check only if the user decides to train one particular
        decision tree object on the data.
        
        Arguments: 
            dataset: pandas.DataFrame. The training dataset.
            depth: The maximum depth that the decision tree is allowed to reach. Default 5.
            useInfoGain: Boolean flag that states whether the user wants the split point quality to be measured
                in terms of Information Gain, as opposed to the default of node "score" (majority vote of splitted labels)
        '''
        
        if depth <= 0 or not isinstance(depth, (int, long)):
            raise mlex.LogicalError("Depth parameter has to be a positive integer")
        self.dataset = dataset
        self.depth = depth 
        self.useInfoGain = useInfoGain
        self.root = None
              
              
    def __str__(self):
        
        '''
        Description: Stringifying method.  Converts the classifier to a string representation.
        Arguments: None
        Return value: A string, presumably with some data about this object.
        '''
        return 'A decision tree classifier of depth ' + self.depth
    
               
    def __selectFeature__(self, trainingData, features):
    
        '''
        Description: Select the best feature found by self.__sortFeatures__
        
        Arguments:
            trainingData: pandas.DataFrame. The training data available
            at the current level of the tree.
            features: a list of features to select from
        
        Return value:
        A tuple (feature, split_point), where feature is the highest
        scoring feature on trainingData, and split_point is the
        split point which yielded that score.
        '''
         
        # Just run self.__sortFeatures__ and then pick the 
        # first value in the returned list.
            
        return self.__sortFeatures__(trainingData, features)[0]
    
    def __sortFeatures__(self, trainingData, features):
    
        '''
        Description: Similar to selectFeature, 
        only difference being that the features are
        stored in a list alongside their scores, and then 
        the list is sorted according to the scores. 
    
        
        Arguments:
            trainingData: pandas.DataFrame. The training data available
            at the current level of the tree.
            features: a list of features to sort in descending score order
            useInfoGain: a flag, by default False, which indicates whether information gain
                is the default split point quality measure (as opposed to the default of score)
        
        Return value:
        A sorted list of tuples (feature, split_point). 
        '''
        
        featureList = []                                        # the tuple list that we will return 
        for featureName in features:                            # loop through all features
            if featureName == 'spam':                           # label not a feature!
                continue
            bestFeatureSplit = None                             # initialize best split for this feature
            bestFeatureSplitScore = 0
            tempCopy = copy.deepcopy(trainingData[featureName]) # worried about sorting an argument in place
            tempCopy.sort()
            tempCopy = tempCopy[::-1].dropna()                  # I prefer descending order and I don't consider null values
            for i in range(tempCopy.index.size):                # loop through all examples in pairs
                if i == tempCopy.index.size -1:                 # avoid memory leak when encountering last example
                    break                  
                        
                # Important: Note the difference between
                # indexing the index and indexing the data
                             
                exampleIndex1 = tempCopy.index[i]
                exampleIndex2 = tempCopy.index[i + 1]           # retrieve the indices of the two examples
                example1 = trainingData.ix[exampleIndex1, :]    # and the two examples themselves
                example2 = trainingData.ix[exampleIndex2, :]
        
                # if the two adjacent examples differ in BOTH label
                # and feature value, calculate the score for the two sub-datasets
                # If you need to use information gain, add "True" as the second argument
                # to self.__calculateScore__.
                
                if(example1['spam'] != example2['spam'] and 
                   example1[featureName] != example2[featureName]):
                    if self.useInfoGain == False:
                        
                        # Use score for current node 
                        
                        scoreLarger = self.__calculateScore__(trainingData[trainingData[featureName] >= 
                                                                  example2[featureName]])
                        scoreSmaller = self.__calculateScore__(trainingData[trainingData[featureName] < 
                                                                  example2[featureName]])
                        totalScore = scoreLarger + scoreSmaller
                    else:
                        
                        # Use Information Gain for current node
            
                        # First, compute cardinalities to weight info gain appropriately
                        
                        cardinalityLarger = trainingData[trainingData[featureName] >= example2[featureName]].shape[0]
                        cardinalitySmaller = trainingData[trainingData[featureName] < example2[featureName]].shape[0]
                        cardinalityCurrent = trainingData.shape[0]
                        fractionLarger = cardinalityLarger / cardinalityCurrent
                        fractionSmaller = cardinalitySmaller / cardinalityCurrent
                        # Second, compute all entropies
                        
                        currentEntropy = self.__calculateEntropy__(trainingData)
                        entropyLarger =  self.__calculateEntropy__(trainingData[trainingData[featureName] >= example2[featureName]])
                        entropySmaller = self.__calculateEntropy__(trainingData[trainingData[featureName] < example2[featureName]])
                        
                        # Third, compute information gain based on both entropies and cardinalities
                        
                        totalScore = currentEntropy - (fractionLarger* entropyLarger + fractionSmaller* entropySmaller)
                                                
                    if(totalScore > bestFeatureSplitScore):
                        bestFeatureSplitScore = totalScore      # update best split score
                        bestFeatureSplit = example2[featureName]
                    
            # end of for loop that scans through examples
            # we now have data that helps us see whether we improved upon
            # our already existing "best" feature
            
            featureList.append((featureName, bestFeatureSplit, bestFeatureSplitScore))
        
        # end of for loop that scans through features
        
        return sorted(featureList, key=lambda data: data[2])[::-1]
    
    # end of method    
    
    def __calculateScore__(self, dataset):
    
        '''
        Description: Given a continuous feature vector (column)
        in the data and a split point, calculate the quality of
        the splitting. Metric used: score (majority vote of labels on
        either side of the splitting       
        Arguments: 
            dataset: pandas.DataFrame        
        Return value: The score of the feature 
        '''
        
        if dataset.shape[0] == 0:                                          # If there are no rows to calculate a score from, return a score of zero
            return 0       

        if 1 in dataset.groupby('spam').shape.keys():
            positiveExampleCount = dataset.groupby('spam').shape[1][0]
        else:
            positiveExampleCount = 0
        if -1 in dataset.groupby('spam').shape.keys():
            negativeExampleCount = dataset.groupby('spam').shape[-1][0]
        else:
            negativeExampleCount = 0 
        return max([positiveExampleCount, negativeExampleCount])       # python's default max faster than numpy.max for simple lists 
        
            
    def __calculateEntropy__(self, dataset):
        
        '''
        Description: Calculate the entropy of a given dataset. 
        Arguments: 
            dataset: pandas.DataFrame
        Return value: the entropy of the dataset (float)
        '''
        if dataset.shape[0] == 0:
            # raise mlex.DatasetError("Cannot compute the entropy of an empty dataset.")
            return 0                                                    # TODO: check whether this is correct
        if 1 in dataset.groupby('spam').shape.keys():
                positiveExampleCount = dataset.groupby('spam').shape[1][0]
        else:
                positiveExampleCount = 0
        if -1 in dataset.groupby('spam').shape.keys():
                negativeExampleCount = dataset.groupby('spam').shape[-1][0]
        else:
                negativeExampleCount = 0
        
        positivesFraction = positiveExampleCount / dataset.shape[0]     # division library will take care of quotients
        negativesFraction = negativeExampleCount / dataset.shape[0]
        return -positivesFraction * np.log2(positivesFraction) -negativesFraction* np.log2(negativesFraction)
    
    def train(self):
        
        '''
        Description: Train the decision tree on the dataset provided.
            Merely a wrapper for the private method __trainRec__, which implements
            the recursive decision tree training algorithm outlined in CIML.
            Training only takes place if the dataset is not empty.
        Arguments: None
        Return value: None
        '''
        if self.dataset.shape[0] > 0:
            allFeatures = self.dataset.columns[0: self.dataset.columns.size -1]
            self.root = self.__trainRec__(self.dataset, allFeatures.tolist(), self.depth, 1)
        else:
            print "No training data provided."
            return
            
    def __trainRec__(self, dataset, features, depth, currentDepth): # Not to be confused with train wrecks.
        
        '''
        Description: Execute the decision tree training algorithm 
            as described on CIML. Recursive method.
        
        Arguments:
            dataset: the portion of the dataset available at the current tree node.
            features: a list of features to split at
            depth: the maximun depth of the (shallow) decision tree
            
        Return value: Node
        '''
        
        # The first thing that we need to do is ensure that the dataset is not empty. At this point,
        # this can only occur if the parent node made a perfect split and we just happen to be on the
        # losers' part of it. In this case, the caller itself has to be turned into a leaf node, whose 
        # label is the label corresponding to the path other than the one followed to reach the present
        # stack frame.
        
        if dataset.shape[0] == 0:
            raise mlex.DatasetError("Cannot split an empty dataset")
        
        # If you had to make a prediction now, what would it be?
        
        label = self.__majorityLabel__(dataset)
     
        # Termination conditions of the training algorithm:
        #     1) no more features to examine
        #     2) Data is completely unambiguous, so you don't need to split any further
        #     3) Maximum depth of decision tree reached: not allowed to split any further.
        #     4) The dataset is empty because the parent splitting provided us with no data whatsoever
        #        (see exception above for more).
        
        if features == [] or self.__dataUnambiguous__(dataset) or currentDepth == depth: 
            return self.__addNode__(None, None, None, None, True, label)    # no features, split point and children nodes in a leaf node
        
        # Run feature selection algorithm on the current dataset and feature vector
        
        (bestFeature, splitPoint, splitScore) = self.__selectFeature__(dataset, features) # We don't really use splitScore
        
        # split data based on feature
        
        positiveData = dataset[dataset[bestFeature] >= splitPoint]  
        negativeData = dataset[dataset[bestFeature] < splitPoint]
        
        # Remove the feature we just splitted on. 
                          
        featureCopy = copy.deepcopy(features)
        featureCopy.remove(bestFeature)
        
        # Recursive call for both 'positive' and 'negative' splits. 
        
        try:
            left_child = self.__trainRec__(negativeData, featureCopy, depth, currentDepth + 1)
        except mlex.DatasetError: 
            
            # The node itself is redundant and can be deleted. Its positive leaf node child takes its place.
            
            return self.__addNode__(None, None, None, None, True, 1)
            
        try:
            right_child = self.__trainRec__(positiveData, featureCopy, depth, currentDepth + 1)
        except mlex.DatasetError:
            
            # Again, this node is redundant, so we replace it with its negative leaf node.
        
            return self.__addNode__(None,  None, None, None, True, -1)
        
        # creation and return of the decision tree node
        
        return self.__addNode__(bestFeature, splitPoint, left_child, right_child)
    
        # end of method
        
    def __addNode__(self, feature, split_point, left_child, right_child, isLeaf = False, label=None):
        
        '''
        Description: Create and return a new node in the decision tree. The node
        can be either an inner (splitter) node or a leaf node.
        
        Arguments: 
            feature: the feature represented by the node (only makes sense for splitter nodes)
            split_point: the value at which the feature is splitted
            isLeaf: optional boolean flag indicating whether the node is a leaf node
            label: optional labeling of the (leaf) node
            
        Returns: Node
        '''
        return Node(feature, split_point, left_child, right_child, isLeaf, label)
    
    def __majorityLabel__(self, dataset):
        
        '''
        Description: Return the majority label of a given dataset
        Arguments:
            dataset: pandas.DataFrame
        Returns: An integer representing the majority label of dataset.
        '''
        
        # If the caller provides an empty dataset, throw an exception
        
        if dataset.shape[0] == 0:
            raise mlex.DatasetError("Cannot compute majority label from an empty dataset.")
        
        # Calculate and return majority label
        
        if 1 in dataset.groupby('spam').shape.keys():
            positiveLabelCount = dataset.groupby('spam').shape[1][0]
        else:
            positiveLabelCount = 0
        if -1 in dataset.groupby('spam').shape.keys():
            negativeLabelCount = dataset.groupby('spam').shape[-1][0]
        else:
            negativeLabelCount = 0
        if positiveLabelCount !=negativeLabelCount:
            return np.sign(positiveLabelCount - negativeLabelCount) 
        else: 
            return 2 * np.random.randint(0, 2) - 1                          # flip a coin if classes have equal count
                 
                     
    
    def __dataUnambiguous__(self, dataset):
        
        '''
        Description: Estimate whether the provided data is unambiguously labeled, i.e has 
            positive or negative labels only.
        Arguments:
            dataset: pandas.DataFrame
        Returns: True if data unambiguous, False otherwise.
        '''
        # If the caller provides an empty dataset, throw an exception.
        
        if dataset.shape[0] == 0:
            raise mlex.DatasetError("Cannot estimate whether an empty dataset is unambiguous")
        
        # Dataset is unambiguous if either label is absent.
        
        return 1 not in dataset.groupby('spam').shape.keys() or -1 not in dataset.groupby('spam').shape.keys()
    
    
    def classifyWithAllDepths(self, testPoints):
        
        '''
        Description: Classify a set of testing examples using all the different trees that were
                trained with different depths during tuning. Useful for understanding the relationship
                between development (tuning) error and generalization (testing) error. If no tuning
                has taken place, this method informs the user and falls back to self.classify.
        Arguments: testPoints: pandas.DataFrame
        Return Value: None, but plenty of printings.
        '''
        
        if testPoints.shape[0] == 0:
            raise LogicalError("Cannot classify an empty dataset.")
        testPointLabels = testPoints['spam'].values
        self.tests = []                                   # keep track of test data, the same way you do with tune data
        if self.tunings == None:
            print "Dataset hasn't been tuned, performing classification with default depth of " + str(self.depth) + " instead..."
            classifications = self.classify(testPoints)
            testingError = np.mean ( (testPointLabels * classifications) < 0)
            print "Testing error of decision tree is: " + str(testingError)
            self.tests.append((self.depth, testingError))
            return 
        else:
            tempTreeStore = self.root     # Hold the current tree somewhere, because we will be changing the root pointer a lot. 
            for tuningData in self.tunings:
                print "Classifying with depth: " + str(tuningData[0]) + "."
                print "For this depth, the tuning error was: " + str(tuningData[1]) + "."
                self.root = tuningData[2]
                classifications = self.classify(testPoints)
                testingError = np.mean ( (testPointLabels * classifications) < 0)
                print "Testing error of this decision tree is: " + str(testingError)
                self.tests.append((tuningData[0], testingError))
            self.root = tempTreeStore
        print "Estimated training error with all different depths."
        
    def classify(self, testPoints):
        
        '''
        Description: Classify a set of testing examples using the trained decision tree.
            A wrapper for the recursive function self.__classifyRec__, which implements the CIML algorithm.
        Arguments: 
            testPoints: pandas.DataFrame
        Returns:
            The classification of each test point (1 for spam and -1 for not spam)
        '''
        if testPoints.shape[0] == 0:
            print "Please provide an example to test"
            return None
        elif self.root == None:
            print "Decision tree hasn't been trained on data." # I could offer to flip a coin for each testing example,
            return None                                        # but I thought it wasn't worth the effort
        else:                                      
            classifications = []
            for testPointIndex in testPoints.index:
                testPoint = testPoints.ix[testPointIndex]
                classifications.append(self.__classifyRec__(self.root, testPoint))
            return classifications
        
    def __classifyRec__(self, node, testPoint):
        
        '''
        Description: Classify ONE testing example using the recursive algorithm outlined in CIML.
        Arguments: 
            tree: pandas.Series
        Return value: (after recursion) Classification of test point (1 for spam, -1 otherwise)
        '''
        if node.isLeaf == True:
            return node.label
        else:
            if testPoint[node.feature] < node.split_point:
                return self.__classifyRec__(node.left, testPoint)
            else:
                return self.__classifyRec__(node.right, testPoint)
            
            
    def dump(self, filePath):
        
        """
        Description: store the current object to a file. Stolen verbatim from project description. Pickle library used.
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
            
    
    def tune(self, tuningSet, lowerDepthBound = 4, higherDepthBound = 20):
        
        '''
        Description: tune the decision tree by measuring generalization error for various values of depth d.
            By default the values checked are between 4 and 20 inclusive.
        Arguments: 
            tuningSet: pandas.DataFrame. Set of examples to measure generalization error from.
            lowerDepthBound: Integer representing the lowest decision tree depth to use for tuning.
            higherDepthBound: Integer representing the lowest decision tree depth to use for tuning.
        Return Value: None
        '''
        
        # Some argument checks first. We prefer to raise exceptions wherever the documentation of a method
        # explicitly states that the return value is None, so as to not expect of the caller to check an
        # output that is not expected to be present.
        
        if lowerDepthBound > higherDepthBound:
            print "Switching your arguments for lower and higher bound of tree depth..."
            lowerDepthBound, higherDepthBound = higherDepthBound, lowerDepthBound
        
        if not isinstance(lowerDepthBound, (int, long)) or not isinstance(higherDepthBound, (int, long)):
            raise TypeError("Integer bounds required for decision tree depth.")
        
        if(tuningSet.shape[0] == 0):
            raise mlex.DatasetError("Please provide examples to tune decision tree on.")
        
        self.tuningSet = tuningSet
        self.tunings = []                                                       # This list will hold (depth, tuning_error, tree root) tuples
        tuningError = np.empty(higherDepthBound - lowerDepthBound + 1)
        tuningSetLabels = tuningSet['spam'].values
        for tuningDepth in range(lowerDepthBound, higherDepthBound + 1):            
            self.depth = tuningDepth
            self.train()
            print "Trained decision tree on depth: " + str(self.depth)
            classifications = self.classify(tuningSet)                          # List of type [-1, 1, 1,....]
            
            # Pay attention to the way that the "tuningError" numpy. ndarray 
            # and the tuningMap dictionary are indexed and filled with values: 
            # The first is offset by "lowerDepthBound" whereas the second is 
            # offset by zero.
             
            tuningError[tuningDepth - lowerDepthBound] = np.mean((classifications * tuningSetLabels) < 0) # product = -1 means wrong classification
            
            # We need to retain data for every tuning, both for statistical purposes and also for the time
            # when we will want to test the data trained with a particular depth. So, for every tuning iteration,
            # we need the following data stored:
            #
            #    a) The value of the hyper-parameter (tree depth)
            #    b) The tuning error (used for statistical purposes)
            #    c) The root of the trained tree, so that we can classify 
            #        test data without having to re-train the tree. We should make
            #        a "deep copy" of this tree so that the entire tree is stored.
            #        This, in essence, turns the current "DecisionTree" object into
            #        a storage space of multiple decision trees, each trained on a different
            #        value of the "depth" hyper-parameter.
            
            tunedDepth, tunedError, trainedTree = self.depth, tuningError[self.depth - lowerDepthBound], copy.deepcopy(self.root)
            self.tunings.append((tunedDepth, tunedError, trainedTree)) 
            print "Classified tuning examples. Tuning error was: " + str(tuningError[tuningDepth - lowerDepthBound])
            
            # Re-initialize tree for the next iteration by setting root to None. This will cause a memory leak
            # for the tree that was currently trained; explicitly run the garbage collector to minimize the time 
            # that this leak is alive.
            
            self.root = None
            gc.collect()
            
    
        # Now that tuning has taken place, set depth to the depth that caused the 
        # least classification error and make the root of the current object
        # point to the relevant tree, so that the object is ready to classify
        # testing points based on the best prediction found in tuning.
        
        self.depth = np.argmin(tuningError) + lowerDepthBound
        for data in self.tunings:
            if data[0] == self.depth:
                self.optimalTuningError = data[1]
                self.root = data[2]
                break
        
        # Some printings to add user-friendliness:
            
        print "Tuned decision tree. Found an optimal depth of " + str(self.depth) + " with a tuning error of " + str(self.optimalTuningError)

            