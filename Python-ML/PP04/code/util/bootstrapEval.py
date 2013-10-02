from __future__ import division
'''
Created on Nov 12, 2012

@author: jason

A set of functions that jointly implement bootstrapped F-score and
ensemble learning with bagging of decision trees (random or non-random).
'''

import numpy as np
import pandas as pd
import pickle as pkl
import inspect
from mlExceptions import DatasetError, LogicalError
from dectree.DTree import get_tree

def bootStrapEval(trueLabs, predictedLabs, K, bag = False, dataset = None):
    
    '''
    This method has two different usages, which are disambiguated between by its boolean
    4th argument, "bag".
     
        1) If bag = false (the default), bootstrap evaluation is implemented. The method takes
        K different bootstrapped samples of its first two arguments (true and predicted labels)
        and computes the F-score for each sample, which it then stores in a list. At the end of
        the execution, it returns the mean and the variance of those F-scores. This is an encoding
        of algorithm 10 at page 65 of CIML.
    
        2) If bag = true, ensemble learning via bagging is implemented. This effectively draws
        K bootstrapped samples (i.e samples with replacement) from the dataset pointed to by 
        the 5th argument and for each sample it trains a decision tree classifier. Every tree
        is stored in a list, which is returned to the caller at the end of execution.
    
     
    
    @param trueLabs: If bag == false, a list of length N representing the true labels of our data. None otherwise.
    @param predictedLabs: If bag == false, a list of length N representing the predicted labels of our data. None otherwise.
    @param K: If bag == false, the number of folds to perform over the labels. Otherwise, the number of 
            bootstrapped samples to draw from the training data.
    @param bag: boolean flag. If false (the default), the method performs bootstrap resampling. If true,
            the method performs bagging of decision trees.
    @param dataset: by default, None. If bag == true,  must be non-None (this is checked for), and is 
            a reference to a pandas.DataFrame which holds the training data to draw samples from.
    @return: If bag == false, mean and standard deviation of "K" - many F-scores. Otherwise, list of
            trained decision tree classifiers.
    @raise LogicalError: If there is some inconsistency, among numerous possible, with respect to the
            arguments provided in each case.
    '''
    
    # Because this method is quite complex, we need to make sure that 
    # the arguments provided to it are consistent with the context 
    # in which we want to use it. We therefore need to do some
    # sanity checking.
    
    if K == None or K == 0: # this is applicable in both usage contexts: we need K > 0
        raise LogicalError, "Method %s: Please provide a positive integer for the K parameter." % inspect.stack()[0][3]
    
    if bag== False: # need to check the validity of the two first arguments
        if trueLabs == None or predictedLabs == None or len(trueLabs) == 0 or len(predictedLabs) == 0:
            raise LogicalError, "Method %s: Cannot compute bootsrapped F-score without true or predicted labels." %  inspect.stack()[0][3]
        if len(trueLabs) != len(predictedLabs):
            raise LogicalError, "Method %s: Mismatch between amount of true and predicted labels." %  inspect.stack()[0][3]
    else:   # need to check the validity of the last argument
        if dataset is None or dataset.shape[0] == 0:
            raise DatasetError, "Method %s: Caller provided a null or empty dataset." % inspect.stack()[0][3]
    
    # Case 1: Bootstrap Resampling
    
    if bag == False:
        
        # Initialize algorithm
    
        scores = list()             # a list of F-scores, initially empty
        numExamples = len(trueLabs)
        
        # For every fold
        
        for _i in range(K):
            foldTrueLabels = list()
            foldPredictedLabels = list()
            
            # For every example
            
            for _j in range(numExamples):
                
                # retrieve and store true and predicted label of example
                
                sampledExampleIndex = np.random.randint(numExamples)        # sample a random example from 0 up to N - 1
                foldTrueLabels.append(trueLabs[sampledExampleIndex])
                foldPredictedLabels.append(predictedLabs[sampledExampleIndex])
            
            # Compute and store the F score for the current fold.
             
            scores.append(__computeFScore__(foldTrueLabels, foldPredictedLabels))
            
        # Return mean and standard deviation of all F scores.
        
        return np.mean(scores), np.std(scores)
    
    # Case 2: Bagging of decision trees
    
    else:
        
        nexamples = dataset.shape[0]
        
        # keep a list of all the decision tree classifiers 
        # that we will train
        
        DTreeList = list()
        
        # for every sample
        
        for datasetSample in range(K):
            
            # keep a list of every example that you sample.
            # In Python terms, this is a list of Series, and
            # we will convert it to a pandas.DataFrame after we
            # complete our inner loop.
            
            examplesInSample = list()
            
            # Select N examples for our sub-dataset
            # by sampling with replacement.
            
            for _example in range(nexamples):
                selectedExample = np.random.randint(0, nexamples)
                examplesInSample.append(dataset.irow(selectedExample))       
            
            subDataset = pd.DataFrame(examplesInSample)
            subDataset.index = np.arange(subDataset.shape[0])
            
            # Train a decision tree classifier on the bootstrapped data
            # and store it in a list.
            print "Building random tree %d." %(datasetSample + 1)
            tree = get_tree(subDataset, 'isgood')
            #print "Tree number %d has an optimal depth of: %d" %(datasetSample+1, tree.optimal_depth)
            DTreeList.append(tree)
        
        # end for _datasetSample    
           
        return DTreeList        
                
    # end function   
    
    
def computeConfidenceInterval(mean, stddev, othermean):
    '''
    Computes the probability according to which the superior performance of the 
    algorithm which yielded an F-score of mean "mean" and standard deviation "stddev"
    is not due to chance. F-scores are assumed to be distributed according to 
    a gaussian distribution.
    
    @param mean: The mean of the superior F-score.
    @param stddev: The standard deviation of the superior F-score.
    @param othermean: The inferior mean F-score.
    @return a float representing the probability that the superior performance of the 
    algorithm which yielded an F-score equal to "mean" is not due to chance.
    @raise LogicalError: If mean < othermean.
    '''
    
    if mean < othermean:
        raise LogicalError, "Method %s: The F-measure questioned is not superior to the other algorithm's F-measure." %(inspect.stack()[0][3])
    
    # If mew - stdev < othermean < mew + stddev, we have a probability of 68.2%.
    if mean - stddev < othermean < mean + stddev:
        return 0.682
    # If mew - 2stddev < othermean < mew + 2stddev, we have a probability of 95.4%.
    elif mean - 2 * stddev < othermean < mean + 2 * stddev:
        return 0.954
    # If mew - 3stddev < othermean < mew + 3stddev, we have a probability of 99.7%.
    elif mean - 3 * stddev < othermean < mean + 3 * stddev:
        return 0.997
    else:
        return 1
    
def __computeFScore__(trueLabels, predictedLabels):
    
    '''
    Compute the F-Score of the algorithm that produced the predictedLabels argument.
    
    @param trueLabels: a list corresponding to the true labels of a dataset. 
    @param predictedLabels: a list corresponding to the predicted labels of our learning algorithm.
    @return: the F-score computed by comparing the two labels element-wise.
    @raise  LogicalError: if the provided lists are None or empty or if the two lists are of mismatch length.
    '''
    
    # Sanity checking
     
    if trueLabels == None or predictedLabels == None or len(trueLabels) == 0 or len(predictedLabels) == 0:
        raise LogicalError, "Method %s: provided empty or None lists." % inspect.stack()[0][3]
    if len(trueLabels) != len(predictedLabels):
        raise LogicalError, "Method %s: mismatch length between true labels and predicted labels." % inspect.stack()[0][3]
    
    
    # Initialization of variables
    
    TP = FP = FN = 0
    
    # For every predicted label
    for i in range(len(predictedLabels)):
        
        # Compare it to the true label and increment respective variable.
        if trueLabels[i] == -1 or trueLabels[i] == 0:   # Both 0 and -1 acceptable as a negative label
            if predictedLabels[i] == 1:                 # False positive
                FP+=1
        else:                                           # positive true label
            if predictedLabels[i] == 1:                 # True Positive
                TP+=1                                
            else:                                       # False Negative
                FN+=1
    
    # Compute Precision, Recall, F-measure, and return F-measure.
    
    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1Score = 2 * (Recall * Precision) / (Recall + Precision)
    return F1Score 



def storeFScores():
    
    # load the prediction-ready training and testing data
        
    trainData = pd.load('proc_data/prediction_ready_trainDat.pda')
    testData = pd.load('proc_data/prediction_ready_testDat.pda')
    trainData = trainData.dropna()
    testData = testData.dropna()
        
    # To answer question 5, we need to perform binary classification on 7 different sub-datasets.
    # We will use Hector's decision trees, and for each case we will get both the true and the predicted ratings
    # and extract the bootstrapped F-score. As a beginning experiment, we will consider K = 10 folds
    # per bootstrap evaluation.
        
    FScoreMeansAndVars = list()             # a list of (mean, stdev) tuples that might prove handy
        
    # We have 7 different feature vectors for both training and testing data 
    # that we need to consider, which we now put in the following two lists.
        
    trainFeatVecs = [
                        trainData.ix[:, :23],                                    # content-based system (user + item features)
                        trainData.ix[:, (22, 24)],                               # using only item-item predicted ratings
                        trainData.ix[:, (22, 23)],                               # using only user-user predicted ratings
                        trainData.ix[:, 22:25],                                  # using both item-item and user-user predicted ratings
                        trainData.ix[:, :23].join(trainData.ix[:, 24:25]),     # content-based + item-item predicted ratings
                        trainData.ix[:, :23].join(trainData.ix[:, 23:24]),     # content-based + user-user predicted ratings
                        trainData.ix[:, :23].join(trainData.ix[:, 23:25])      # content-based + both predicted ratings
                            ]
        
    testFeatVecs = [
                            testData.ix[:, :23],                                     # content-based system (user + item features)
                            testData.ix[:, (22, 24)],                                # using only item-item predicted ratings
                            testData.ix[:, (22, 23)],                                # using only user-user predicted ratings
                            testData.ix[:, 22:25],                                   # using both item-item and user-user predicted ratings
                            testData.ix[:, :23].join(testData.ix[:, 24:25]),       # content-based + item-item predicted ratings
                            testData.ix[:, :23].join(testData.ix[:, 23:24]),       # content-based + user-user predicted ratings
                            testData.ix[:, :23].join(testData.ix[:, 23:25])        # content-based + both predicted ratings
                            ]
        
        # Now that we have all 7 different training and testing datasets,
        # we can compute the bootstrapped F-score for every setup.
        # We will store these bootstrapped F-scores in a list, which we will
        # then store on disk for easy future access. We will
        # use K=100 folds for our experiments.
        
    for i in range(len(trainFeatVecs)):
            
        print "Training decision tree for configuration %d." %(i+1)
        tree = get_tree(trainFeatVecs[i], 'isgood')
        print "Trained a decision tree, found an optimal depth of: %d" %(tree.optimal_depth)
        print "Getting predictions of decision tree on testing data."
        predictions = tree.predict(testFeatVecs[i])
            
        print "Computing bootstrapped F-score."
        mean, stddev = bootStrapEval(testFeatVecs[i]['isgood'].values, predictions, 1000)
        print "Computed a mean F-score of %.4f with a std. dev. of %.4f." %(mean, stddev)
            
        print "Storing bootstrapped F-score of configuration %d in a list." %(i+1)
        FScoreMeansAndVars.append((mean, stddev))
       
    print "Storing all F-scores on disk."
    fp = open('proc_data/bootstrappedFScores.pda', 'wb')
    pkl.dump(FScoreMeansAndVars, fp)
    fp.close()

def printConfidences(FScoreList):
    """
    Receives a list of (Fscore mean, Fscore stddev) tuples which it then processes sequentially
    and prints the confidences according to which the better measurements are significantly better
    than their counterparts.
    
    @param FScoreList: a list of (FScoreMean, FScoreStdDev) tuples.
    @return: None
    @raise LogicalError: If FScoreList is None or empty. 
    """
    
    if FScoreList is None or len(FScoreList) == 0:
        raise LogicalError, "Method %s: Cannot print the confidences over a null or empty list." %(inspect.stack()[0][3])
    
    # We need to compare the bootstrapped F1 score of model 1 against any other model's,
    # then model 2's with every other model except 1, then 3's with 4, 5, 6, 7 and so on.
    
    for i1 in range(len(FScoreList) - 1):
        for i2 in range(i1 + 1, len(FScoreList)):
            
            # Get the mean F1 Score and the standard deviation of F-scores 
            # for both prediction models.
            
            (mean1, stddev1) = FScoreList[i1]
            (mean2, stddev2) = FScoreList[i2]
            
            # Find the model that does better.
            
            maxMean, minMean = max(mean1, mean2), min(mean1, mean2)
            stddev = stddev1 if maxMean == mean1 else stddev2
            betterPredictor = i1 + 1 if maxMean == mean1 else i2 + 1
            
            # Compute the confidence that the better performance of the
            # model was not by chance and print it.
            
            confidence= computeConfidenceInterval(maxMean, stddev, minMean)
            print "We compared models %d and %d." %(i1 + 1, i2 + 1)
            print "There is a %1.f%% chance that the superior performance of model %d was not by chance." %(100*confidence, betterPredictor)
            
        #end inner for loop
    # end outer for loop
            
                
import os

if __name__ == "__main__":
    
    try:
        os.chdir("../../")
        np.random.seed(1)               # change this after debugging
        ###### PART 1: Training ############
        
        trainDat = pd.load('proc_data/prediction_ready_trainDat.pda')
        testDat = pd.load('proc_data/prediction_ready_testDat.pda')      
        #classifierList = bootStrapEval(None, None, 50, True, trainDat) # we will build 50 trees of depth 20.
        
        # dump it to disk, once again
        #print "We bagged %d classifiers, which we will now store on disk." %(len(classifierList))
        #fp = open('proc_data/baggedRandomClassifiers.pda', 'wb')
        #pkl.dump(classifierList, fp)
        #fp.close()       
        
        #print "\n######## We are done with the training phase. We will now proceed with the evaluation phase.########\n"
        ##### PART 2: Evaluation ##############
        
        fp1 = open('proc_data/baggedRandomClassifiers.pda', 'rb')
        fp2 = open('proc_data/baggedClassifiers.pda', 'rb')
        randomForest = pkl.load(fp1)
        baggedTrees = pkl.load(fp2)
        fp1.close()
        fp2.close()
        
        # Get predictions for both the random forest and bagged classifiers.
        
        randomForestPredictionMatrix = np.array([tree.predict(testDat) for tree in randomForest]).transpose()
        baggedPredictionMatrix = np.array([tree.predict(testDat) for tree in baggedTrees]).transpose()
        
        # Get the voted labels for both the random forest and our bagged classifiers.
        randomForestVotedLabels = [1 if randomForestPredictionMatrix[j].tolist().count(1) > randomForestPredictionMatrix[j].tolist().count(-1) else -1 for j in range(randomForestPredictionMatrix.shape[0])]
        baggedTreesVotedLabels = [1 if baggedPredictionMatrix[j].tolist().count(1) > baggedPredictionMatrix[j].tolist().count(-1) else -1 for j in range(baggedPredictionMatrix.shape[0])]
        
        # Compute bagging's and random forest's mean F-score over 1000 folds 
        
        (bagmean, bagstddev) = bootStrapEval(testDat['isgood'].values, baggedTreesVotedLabels, 1000)
        print "The mean F-score of our bagged trees over %d folds of the testing data was %.3f, and the standard deviation was %.3f." %(1000, bagmean, bagstddev)
        
        (rfmean, rfstddev) = bootStrapEval(testDat['isgood'].values, randomForestVotedLabels, 1000)
        print "The mean F-score of our random forest over %d folds of the testing data was %.3f, and the standard deviation was %.3f." %(1000, rfmean, rfstddev)
        
        # Load the bootstrapped F-score of sub-question VII
        
        fp = open('proc_data/bootstrappedFScores_depth20.pda')
        bootstrappedFScores = pkl.load(fp)
        fp.close()
        (q5mean, q5stddev) = bootstrappedFScores[6]             
        print "In comparison, the bootstrapped F-score for sub-question VII of question 5 was %.3f and its standard deviation was %.3f." %(q5mean, q5stddev)
        
        # Compare all the algorithms through their confidence intervals.
        
        printConfidences([(rfmean, rfstddev), (bagmean, bagstddev), (q5mean, q5stddev)])
        
        print "All done, exiting."
        quit()    
            
    except DatasetError as e:
        print "A dataset - related error occurred: " + str(e)
        quit()
    except LogicalError as l:
        print "A logical error occurred: " + str(l)
        quit()
    except Exception as e:
        print "An exception occurred: " + str(e)
        quit()
    

