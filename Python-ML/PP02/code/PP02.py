from __future__ import division

'''
Created on Oct 11, 2012

@author: Jason
'''

from avgperceptron import AveragedPerceptron
from OVA import OVA
from mlExceptions import DatasetError, LogicalError
import numpy as np
import pickle as pk
import os, sys
import util
import copy
from dummy_thread import exit


def getRandomLabeledExamples(label, numExamples, data):
    
    '''
    ' This method receives a dataset as input and returns "numExamples" random examples 
    ' of class "label". If the examples of a label are less than "numExamples", then 
    ' the entire example sequence is returned, in shuffled order.
    '''
    
    examplesOfClass = copy.deepcopy([examples for examples in data if examples[-1] == label])
    if len(examplesOfClass) <= numExamples:
        np.random.shuffle(examplesOfClass)
        return examplesOfClass              
    else:
        # Create a dataset with "numExamples" random examples and return it.
        selectedExamples = []
        np.random.seed(1)
        for _i_ in range(numExamples):
            randomIndex = np.random.randint(0, len(examplesOfClass))
            selectedExamples.append(examplesOfClass[randomIndex])
        return selectedExamples
    
def getReducedDataset(labelList, randomExPerLabel, dataset):
    
    '''
    ' Use the getRandomLabeledExamples() method to
    ' get a random number of examples per label and then concatenate all
    ' those random examples into a dataset which has randomExPerLabel * |labelList| (e.g. 5 * 101 = 505 for Caltech)
    ' examples.
    '''
    reducedDataset = []
    for label in labelList:
        reducedDataset = reducedDataset + getRandomLabeledExamples(label, randomExPerLabel, dataset)
    np.random.shuffle(reducedDataset)
    return reducedDataset
    
def main():
    
#    # Part 1: Load data
#    
#    DIR_categories=os.listdir('../input_data/training/');      # list and store all categories (classes)
#    allFeatures=[]                                             # this list will store all training examples 
#    imLabels=[]                                               # this list will store all labels
#    labelCount=0;                                                      # this integer will effectively be the class label in our code (i = 1 to 101)
#    labelNames = []                                           # we will need the label names in order to plot their cardinalities later on
#    labelCardinalities = []                                   # see above
#    for cat in DIR_categories:                                # loop through all categories
#        if os.path.isdir('../input_data/training/'+ cat):   
#            labelNames.append(cat)
#            labelCount=labelCount+1;                                             # i = current class label
#            count = 0
#        
#            DIR_image=os.listdir('../input_data/training/'+ cat +'/');      # store all images of category "cat" 
#            for im in DIR_image:                                           # loop through all images of the current category
#                if (not '._image_' in im):                                 # protect ourselves against those pesky Mac OS X - generated files
#                    F = np.genfromtxt('../input_data/training/'+cat+'/'+im, delimiter=' '); # F is now an 2-D numpy ndarray holding all features of an image
#                    F = np.reshape(F,21*28);                               # F is now a 588 - sized 1-D ndarray holding all features of the image
#                    F = F.tolist();                                        # listify the vector
#                    F.append(labelCount)                                   # we'd like to store the label alongside the example
#                    count = count + 1
#                    allFeatures.append(F);                                  # store the vector
#                    imLabels.append(labelCount);                                    # store the label
#                    labelCardinalities.append(count)
#    print "training data loaded!"
    
    # Store some data on disk so we don't have to 
    # re-read it every time.
#    print " We will now count the counts of all classes to see whether something's wrong with them"
#    exIndex = 1
#    for label in range(1, labelCount + 1):
#        examplesOfLabel = [examples for examples in allFeatures if examples[-1] == label]
#        print "There are: " + str(len(examplesOfLabel)) + " examples of class " + str(label)
#     
#    print "We will now exit"
#    exit()
#    try: 
#        fp = open("../proc_data/trainingData.pdat",'wb')
#        pk.dump(allFeatures, fp)
#        fp.close()
#    except Exception as e:
#        print 'Pickling failed for object allFeatures: Exception: ' + e.message
#        exit()
#        
#    print "Training data stored on disk"
#    
#    try: 
#        fp = open("../proc_data/labelCount.pdat",'wb')
#        pk.dump(labelCount, fp)
#        fp.close()
#    except Exception as e:
#        print 'Pickling failed for object labelCount: Exception: ' + e.message
#        exit()
#    
#    print "Label count stored on disk"
#    
#    # Part 2: Initialize OVA structure and classifiers in memory
#    
#    
#        # First of all we need to draw training and tuning
#        # data from our original Caltech data.
#        # Reminder: testing (development) data has already been made available
#        # to us, so we don't need to partition the original data any further.
     
#        allFeatures = util.load("../proc_data/trainingData.pdat")
#        labelCount = util.load("../proc_data/labelCount.pdat")    
#        numTrainExamples = int(np.floor(.8 * len(allFeatures)))                      # need to convert ndarray scalar to int
#        np.random.seed(1)
#        np.random.shuffle(allFeatures)                                               # this achieves a degree of randomness
#    
#        trainingData = allFeatures[:numTrainExamples]                                # pull training data
#        tuningData = allFeatures[numTrainExamples:]                                  # pull tuning data
#        
##        print "Now that we have the training data in our hands, we will count the cardinalities of class within it: "
##        for label in range(1, labelCount + 1):
##            examplesOfLabel = [examples for examples in trainingData if examples[-1] == label]
##            print "There are: " + str(len(examplesOfLabel)) + " examples of class " + str(label)
#     
#
#        # Once we're done with data, we need to define the 
#        # OVA object in memory and add 101 classifiers inside it.
#        
#        ovaStructure = OVA(trainingData, tuningData, labelCount)                                
#        for _ in range(labelCount):
#            ovaStructure.addClassifier(AveragedPerceptron(5))            # training those classifiers for maxiter = 5
#        
#        print "Created an " + str(ovaStructure)
#    
#        ovaStructure.dump('../proc_data/stored_classifiers/firstOVA_untuned.pdat')
#        print "OVA object dumped in disk."
        
    # Part 3: Tune all classifiers and store the OVA object in memory.
    
        #ovaStructure = util.load("../proc_data/stored_classifiers/firstOVA_untuned.pdat")
        #print "Resumed the following OVA object: " + str(ovaStructure) + "."
        # ovaStructure.printInfo()                                                    # a debugging method that prints some stuff
        #ovaStructure.tune()
        #ovaStructure.dump("../proc_data/stored_classifiers/firstOVA_tuned.pdat")
        #print "We tuned all classifiers of the OVA object and stored them in memory." 
    
    
    
    #Part 4: Test the trained classifiers on the Caltech 101 development data.
    
        # The first thing we need to do is read the development data in memory.
        # We will use the same logic we used to scan the training data.
    
#        validationClasses=os.listdir('../input_data/validation/');      # list and store all categories (classes)
#        validationData=[]                                             # this list will store all training examples
#        label = 0 
#        for cat in validationClasses:                                # loop through all categories
#            if os.path.isdir('../input_data/validation/'+ cat):   
#                DIR_image=os.listdir('../input_data/validation/'+ cat +'/');      # store all images of category "cat"
#                label = label + 1 
#                for im in DIR_image:                                           # loop through all images of the current category
#                    if (not '._image_' in im):                                 # protect ourselves against those pesky Mac OS X - generated files
#                        F = np.genfromtxt('../input_data/validation/'+cat+'/'+im, delimiter=' '); # F is now an 2-D numpy ndarray holding all features of an image
#                        F = np.reshape(F,21*28);                               # F is now a 588 - sized 1-D ndarray holding all features of the image
#                        F = F.tolist();                                        # listify the vector
#                        F.append(label)                                   # we'd like to store the label alongside the example
#                        validationData.append(F);                                  # store the vector
#        print "Validation data loaded!"
#    
#        # We would like to have this representation of data stored in our hard disk
#        # so that we don't have to read it each and every time
#        
#        fp = open("../proc_data/validationData.pdat",'wb')
#        fp2 = open("../proc_data/labelCount.pdat",'wb')
#        pk.dump(validationData, fp)
#        pk.dump(label, fp2)
#        fp2.close()
#        fp.close()
#        fp2.close()
#
#        print "Validation data stored on disk."
        
        # In order to adhere to the project's specifications, we need to 
        # train the OVA scheme in 5, 10, 20, 30, 40, 50, 60 random images per category
        # and then test it against the validation data.
        
        # To do this, we simply need to train 7 different OVA objects, which means 7 * 101 Averaged Perceptrons,
        # and test the accuracy of each against the validation data. We will use the getRandomLabeledExamples()
        # method to retrieve the random examples required, and then we will train our OVA
        
    try:   
#        labelCount = util.load("../proc_data/labelCount.pdat")
#        trainingData = util.load("../proc_data/trainingData.pdat")
#        validationData = util.load("../proc_data/validationData.pdat")
#        accuracy = []
#        for exampleNums in [5, 10, 20, 30, 40, 50, 60]:
#            reducedTrainingData = getReducedDataset(range(1, labelCount + 1), exampleNums, trainingData)
#            ovaClassifier = OVA(reducedTrainingData, None, labelCount)                 # No tuning data provided because we don't need to (2nd argument is None)
#
#            for _label_ in range(labelCount):
#                ovaClassifier.addClassifier(AveragedPerceptron())               # default AveragedPerceptron class MaxIter hyper-parameter for training without having tuned first: 15
#            print "Training " + str(ovaClassifier)
#            ovaClassifier.train()
#            print "Testing " + str(ovaClassifier) + " on validation data."
#            accuracy.append(1.0 - ovaClassifier.test(validationData))           # OVA.test returns error rate, so we subtract that from 1 to retrieve accuracy
#            
#        
#        # We will use drawError() to draw the accuracy.
#        print "Drawing accuracy results"
#        util.drawError([5, 10, 20, 30, 40, 50, 60], accuracy, "Learning curve for multi-class Averaged Perceptron.")
#        
#        # We will store the accuracy for future reference and plotting
#        
#        print "Dumping accuracy results to disk."
#        acFP = open("../proc_data/learningCurve.pdat","wb")
#        pk.dump(accuracy, acFP)
#        acFP.close()
#        
#        print "We stored the accuracy on disk."
#        print "Exiting..."
            
        # Q 7 : Learn the Perceptron with a varying number of iterations
        #
        #
        
        labelCount = util.load("../proc_data/labelCount.pdat")
        trainingData = util.load("../proc_data/trainingData.pdat")
        validationData = util.load("../proc_data/validationData.pdat")
        accuracy = []
        for maxIterVal in [1,10,50,100, 500]:                                                               # not going over 100, took too much time
            reducedTrainingData = getReducedDataset(range(1, labelCount+1), 50, trainingData)                # get 50 examples per class
            ovaClassifier = OVA(reducedTrainingData, None, labelCount)
            for _label_ in range(labelCount):
                ovaClassifier.addClassifier(AveragedPerceptron())
            ovaClassifier.setAllHyperparams(maxIterVal)                                                      # brute-force the perceptrons in this case
            print "Training " + str(ovaClassifier)
            ovaClassifier.train()
            print "Testing " + str(ovaClassifier) + " on validation data."
            accuracy.append(1.0 - ovaClassifier.test(validationData))           # OVA.test returns error rate, so we subtract that from 1 to retrieve accuracy
            
        # We will use drawError() to draw the accuracy.
        print "Drawing accuracy results"
        util.drawSimplePlot([1,10,50,100, 500], accuracy, "Accuracy per maxIter for the Averaged Perceptron", "maxIter value", "Accuracy")
        
        # We will store the accuracy for future reference and plotting
        
        print "Dumping accuracy results to disk."
        acFP = open("../proc_data/accPerMaxIter.pdat","wb")
        pk.dump(accuracy, acFP)
        acFP.close()
        
        print "We stored the accuracy on disk."
        print "Exiting..."
            
    except DatasetError as d:
        print "A dataset-related error occured: " + str(d) + "."
        exit() 
    except LogicalError as l:
        print "A  logical error occured: " + str(l) + "."
        exit()
    except Exception as exc:
        print "An exception occurred: " + str(exc) + "."
        exit()
    except:
        print "An unknown error occurred."
        exit()
        
if __name__ == '__main__':
    main()