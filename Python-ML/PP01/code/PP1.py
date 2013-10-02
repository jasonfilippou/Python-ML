'''
Created on Sep 21, 2012

@author: Jason
'''

import numpy as np
import pandas as pd
from DecTree import DecisionTree
from knn import KNN
from util import load
import DecTree


def main():
    
    #############################################
    # Set up the data as per the first Practicum
    #############################################
    
    spam_values = np.genfromtxt('../input_data/spambase.data', delimiter=',')
    fl = open('../input_data/spambase.names', 'r')
    lines = [line.strip() for line in fl] # J : strip from beginning and ending whitespace
    fl.close()
    
    colnames = [line.partition(':')[0] for line in lines if not (len(line) == 0 or line[0] == '|' or line[0] == '1')]
    colnames.append('spam')
    
    spam_df = pd.DataFrame(spam_values,columns=colnames)
    spam_df['spam']=2*spam_df['spam']-1
    
    # J: Apparently DataFrame.shape is a list or something and the first cell contains the number of samples in the DataFrame
    nsamples = spam_df.shape[0] 
    ntest = np.floor(.2 * nsamples)
    ntune = np.floor(.1 * nsamples)
    
    # we want to make this reproducible so we seed the random number generator
    np.random.seed(1)
    all_indices = np.arange(nsamples) 
    # J: important to shuffle so that you don't know which portion is training, which is testing and which is tuning data
    np.random.shuffle(all_indices) 
    test_indices = all_indices[:ntest] # J: Get shuffled test indices first
    tune_indices = all_indices[ntest:(ntest+ntune)] # J: tune indices second
    train_indices = all_indices[(ntest+ntune):] # J: train indices (the majority) last
    
    # J : now that the "*indices" arrays have been shuffled, you can actually draw the relevant data through
    # DataFrame.ix. The second argument includes all columns, labels included.
    spam_train = spam_df.ix[train_indices,:]
    spam_tune = spam_df.ix[tune_indices,:]
    spam_test = spam_df.ix[test_indices,:]
    
    pd.save(spam_train, '../proc_data/training_data/spam_train.pdat')
    pd.save(spam_tune, '../proc_data/training_data/spam_tune.pdat')
    pd.save(spam_test, '../proc_data/testing_data/spam_test.pdat')
    
    
    #######################################################################
    # See how features are sorted according to their Information Gain score
    #######################################################################
    
    # atestTree = DecisionTree(spam_train, 5, True)
    # print atestTree.__sortFeatures__(spam_train, spam_train.columns)
    
    ###############################################
    #  Training classifiers and saving them on disk
    ###############################################
    
    # Already trained those two, it took about 4 hours total.
    # I've commented them out so that the user doesn't train them
    # by accident. 
     
#    majVoteTree = DecTree.DecisionTree(spam_train, 5, False)
#    print "Tuning a majority vote classifier on all depths between 1 and 15 inclusive."
#    majVoteTree.tune(spam_tune,1, 15)
#    print "Saving this classifier to disk."
#    majVoteTree.dump("../proc_data/dtreeWithMajVote_1_to_15.pyobj")
#    
#    IGTree = DecTree.DecisionTree(spam_train, 5, True)
#    print "Tuning an information gain classifier on all depths between 1 and 15 inclusive."
#    IGTree.tune(spam_tune,1, 15)
#    print "Saving this classifier to disk."
#    IGTree.dump("../proc_data/dtreeWithIG_1_to_15.pyobj")

    # The tuning of the KNN classifier doesn't take as long,
    # but I've commented it out for consistency with the tuning of
    # the decision tree.
    
#    HectorsKNN = KNN(spam_train, spam_train['spam'], 5)
#    print "Tuning Hector's KNN classifier for all values of K between 1 and 41 inclusive:"
#    HectorsKNN.tune(spam_tune, spam_tune['spam'], k=range(1,42,2))
#    print "Saving this classifier to disk."
#    HectorsKNN.dump("../proc_data/HectorsKNN_1_to_41.pyobj") 
    
    ###########################################
    # Playing with stored classifiers
    ###########################################
    
    # Part 1: A decision tree classifier trained with Majority Vote, depths 1 to 10

    print "Loading a decision tree trained with Majority Vote for depths 1 to 10..."
    majVoteTree = load("../proc_data/dtreeWithMajVote_1_to_15.pyobj")
    print "According to the tuning set, the optimal depth for this tree is: " + str(majVoteTree.depth)
    classifications = majVoteTree.classify(spam_test)
    testErrorRate = np.mean ( (spam_test['spam'].values * classifications) < 0)
    print 'For this depth, the error on the test set was %0.3f' % testErrorRate
    print "We will now test all different hyper-parameters found during tuning on the test data:"
    majVoteTree.classifyWithAllDepths(spam_test)
    
    print "\n===========================================================\n"
    
    # Part 2: A decision tree classifier trained with Information Gain, depths 1 to 10
    
    print "Loading a decision tree trained with Information Gain for depths 1 to 10..."
    IGTree = load("../proc_data/dtreeWithIG_1_to_15.pyobj")
    print "According to the tuning set, the optimal depth for this tree is: " + str(IGTree.depth)
    classifications = IGTree.classify(spam_test)
    testErrorRate = np.mean ( (spam_test['spam'].values * classifications) < 0)
    print 'For this depth, the error on the test set was %0.3f' % testErrorRate
    print "We will now test all different hyper-parameters found during tuning on the test data:"
    IGTree.classifyWithAllDepths(spam_test)
    
    print "\n===========================================================\n"
    
    # Part 3: Hector's KNN-classifier
    
    print "Reloading Hector's classifier from disk:"
    HectorsKNN = load("../proc_data/HectorsKNN_1_to_41.pyobj")
    print "According to the tuning set, the optimal K for this classifier is: " + str(HectorsKNN.k) + "."
    classifications = HectorsKNN.classify(spam_test)
    testErrorRate = np.mean ( (spam_test['spam'].values * classifications) < 0)
    print 'For this value of K, the error on the test set was %0.3f' % testErrorRate
    print "We will now test all different hyper-parameters found during tuning on the test data:"
    HectorsKNN.classifyWithAllK(spam_test)
    
    print "\n===========================================================\n"
    
    # Part 4: Weighted Distance KNN
    # No time to implement WDKNN :( :( :(
    
    print "Exiting..."
    
if __name__ == '__main__':
    main()