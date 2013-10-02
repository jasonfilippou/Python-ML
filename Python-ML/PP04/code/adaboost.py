from __future__ import division
'''
Created on Nov 17, 2012

@author: jason
'''

import pandas as pd
import numpy as np
import inspect
from dectree.DTree import get_tree
from util.mlExceptions import *
from util.bootstrapEval import bootStrapEval

def adaboost(trainDat, K):
    
    """
    Implement the ADABoost algorithm, as described in CIML, page 152.
    @param trainDat: a pandas.DataFrame representing our training data.
    @param K: the number of decision tree stumps that we would like to train.
    @return --- a list of K decision tree stumps, trained on weighted data.
            --- a list of K adaptive parameters, used on predictions alongside 
                the individual classifiers' predictions.
    @raise LogicalError if K<= 0, None or not an int
    @raise DatasetError if trainDat is None or empty
    """
    
    if trainDat is None or len(trainDat) == 0:
        raise DatasetError, "Method %s: Cannot train ADAboost on a null or empty dataset." %(inspect.stack()[0][3])
    if K is None or K <= 0 or not isinstance(K, int):
        raise LogicalError, "Method %s: Need to train a positive number of classifiers" %(inspect.stack()[0][3])
    
    print "Starting AdaBoost algorithm."
    # initialize uniform weights
    
    exampleWeights = np.array([(1 / trainDat.shape[0]) for _x_ in range(trainDat.shape[0])])
        
    # run main algorithm 
    classifierList = list()
    adaptParams = list()
    for k in range(K):
        
        # train a decision tree on the weighted training data
        print "Training tree #%d." %(k+1)
        tree = get_tree(trainDat, 'isgood', exampleWeights, 1, 0)  # boost strong learners with aim to predict "rating" column
        classifierList.append(tree)
        # Run predictions on weighted training data 
        print "Getting training data predictions for tree #%d." %(k+1)
        predictions = tree.predict(trainDat)
        
        # Compute training error
        
        trueValues = trainDat['isgood'].values
        
        if len(predictions) != len(trueValues):
            raise LogicalError, "Method %s, model #%d: predictions have to be as many as the true labels." %(inspect.stack()[0][3], k + 1)
        
        # We need a new way to compute training error in the regression case.
        # We will compute the training error as a weighted mean square loss.
        
        squaredDiff = (trueValues - predictions)**2
        print len(squaredDiff)
        squaredDiff *= exampleWeights
        trainingError = np.sum(squaredDiff) / float(len(squaredDiff))
       
        
        # Compute and store the "adaptive" parameter a(k)
        print "training error for tree #%d: %.4f" %(k+1, trainingError)
        currentAdaptParam = 0.5 * np.log((1 - trainingError) / trainingError)
        
        #if type(currentAdaptParam) != float:
            #raise LogicalError, "Method %s, model #%d: type of adaptive parameter was %s instead of float." %(inspect.stack()[0][3], k + 1, type(currentAdaptParam))
        
        adaptParams.append(currentAdaptParam)
        print "Computed adaptive parameter for classifier %d. It is equal to: %.4f" %(k+1, currentAdaptParam)
         
        # Update and normalize example weights
        # Note that this is not a dot product, but an element-wise multiplication.
        
        exponent = -currentAdaptParam *np.array([trueValues[n] for n in range(trainDat.shape[0])])* np.array([predictions[n] for n in range(trainDat.shape[0])])
        
        try:
            len(exponent)
        except TypeError:
            raise LogicalError, "Method %s: \"exponent\" is not an iterable." %(inspect.stack()[0][3]) 
        if len(exponent) != trainDat.shape[0]:
            raise LogicalError, "Method %s: our derivation of \"exponent\" should've yielded a numpy.ndarray of size %d at this point." %(inspect.stack()[0][3], trainDat.shape[0])
        
        multiplier = exampleWeights * np.exp(exponent)
        
        try:
            len(multiplier)
        except TypeError:
            raise LogicalError, "Method %s: \"multiplier\" is not an iterable." %(inspect.stack()[0][3]) 
        
        if len(multiplier) != trainDat.shape[0]:
            raise LogicalError, "Method %s: our derivation of \"multiplier\" should've yielded a numpy.ndarray of size %d at this point." %(inspect.stack()[0][3], trainDat.shape[0])
        
        # Now we need to normalize, and God only knows how we're supposed to do this.
        
        normalizer = np.sum(multiplier)             # TODO: Decide whether this is the correct normalizer    
        exampleWeights = exampleWeights / normalizer   
        
        try:
            len(exampleWeights)
        except TypeError:
            raise LogicalError, "Method %s, model #%d: after the update to \"exampleWeights\", this variable no longer represents a numpy.ndarray." %(inspect.stack()[0][3], k + 1)
        if  len(exampleWeights) != trainDat.shape[0]:
            raise LogicalError, "Method %s, model #%d: the update to exampleWeights should've yielded a numpy.ndarray of size %d at this point." %(inspect.stack()[0][3], k + 1, trainDat.shape[0])
        
    return classifierList, adaptParams

import os
import pickle as pkl

if __name__ == "__main__":
    
    try:
        
        os.chdir("../")
        
        # load training and testing data from disk
        trainData = pd.load('proc_data/prediction_ready_trainDat.pkl')
        testData = pd.load('proc_data/prediction_ready_testDat.pkl')
#        
#        # train boosted classifiers
        print "Training boosted classifiers..."
        #print trainData.columns
        boostedTrees , adaptiveParams = adaboost(trainData, 50)
        print "Trained 50 classifiers and retrieved relevant adaptive parameters. We will now store all this data on disk."
        fp1 = open('proc_data/boostedClassifiers.pda', 'wb')
        fp2 = open('proc_data/adaptiveParams.pda', 'wb')
        pkl.dump(adaptiveParams, fp2)
        pkl.dump(boostedTrees, fp1)
        fp2.close()
        fp1.close()
        
        # Get weighted predictions. Remember that AdaBoost computes predictions
        # by taking the sign of the weighted sum of predictions of every classifier,
        # where the weights of each individual prediction are none other than the
        # corresponding adaptiva parameter of the classifier.
        
#        fp1 = open('proc_data/boostedClassifiers.pda', 'rb')
#        fp2 = open('proc_data/adaptiveParams.pda', 'rb')
#        boostedTrees = pkl.load(fp1)
#        adaptiveParams = pkl.load(fp2)
#        fp2.close()
#        fp1.close()
#        print "Getting predictions..."
#        adaboostPredictionMatrix = np.array([tree.predict(testData) for tree in boostedTrees]).transpose()
#        adaboostVotedLabels = [np.sign(np.sum(adaptiveParams * adaboostPredictionMatrix[j])) for j in range(adaboostPredictionMatrix.shape[0])]
#        
#        if len(adaboostVotedLabels) != testData.shape[0]:
#            raise LogicalError, "Method %s: Predicted labels have to be as many as the testing examples." %(inspect.stack()[0][3])
#        
#        # Compute mean and stddev of F-score over 1000 folds:
#        
#        print "Computing mean and std. dev of F-score over 1000 folds..."
#        (boostmean, booststddev) = bootStrapEval(testData['isgood'].values, adaboostVotedLabels, 1000)
#        
#        print "Mean: %.4f , std. dev: %.4f" %(boostmean, booststddev)
        print "All done, exiting."
        
    except DatasetError as e:
        print "A dataset - related error occurred: " + str(e)
        quit()
    except LogicalError as l:
        print "A logical error occurred: " + str(l)
        quit()
    except Exception as e:
        print "An exception occurred: " + str(e)
        quit()
        
        