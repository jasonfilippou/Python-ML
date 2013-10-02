'''
Created on Dec 9, 2012

@author: jason
'''
from adaboost import adaboost
import pickle as pkl
import numpy as np
import pandas as pd
from cfiltering import *
from cfiltering_item import *

class RatingPredictor(object):
    '''
    A class that will be used for predictions
    in the final semester challenge.
    '''
    
    def __init__(self, data):
        
        '''
        Initialize our predictor.
         @param df a pandas.DataFrame which *does* contain the column that we want to eventually
            predict, so that the decision tree code may attain tuning.
        '''
        self.trainDat = data
    
    def train(self):
        
        '''
        Adaboost, where 75 strong classifiers (decision trees with maxdepth = 10) are boosted,
        will be used.
        '''
        self.boostedTrees, self.adaptiveParams = adaboost(self.trainDat, 200)
        fp1 = open('proc_data/boostedTreesFinalChal.pkl', 'wb')
        fp2 = open('proc_data/adaptiveParamsFinalChal.pkl', 'wb')
        pkl.dump(self.boostedTrees, fp1)
        pkl.dump(self.adaptiveParams, fp2)
        fp1.close()
        fp2.close()
        
    def predict(self, df):
        
        """
        Predict movie ratings for dataframe df. Uses the classifiers and the adaptive parameters
        stored by adaboost to predict every example as a sign of a weighted sum of predictions. 
        
        Arguments:
        df: a pandas DataFrame with the same features as the data in PA03 (it may not contain the isgood column so you shouldn't refer to it)
        
        Returns:
        Predictions (between 1-5) for each of the user-movie pairs in df
        """
       
        if 'isgood' in df.columns.values:
            df.pop('isgood')
        
        #trueLabs = df['rating'].values
        print "Getting predictions..."
        adaboostPredictionMatrix = np.array([tree.predict(df) for tree in self.boostedTrees]).transpose()
        adaboostVotedLabels = [np.sign(np.sum(self.adaptiveParams * adaboostPredictionMatrix[j])) for j in range(adaboostPredictionMatrix.shape[0])]
        return adaboostVotedLabels
    
def rating_class(df, ratings):
    """
    Build a movie rating predictor
    
    
    Arguments:
    df: a pandas DataFrame with the same features as the data in PA03 (it may not contain the isgood column so you shouldn't refer to it) without the ratings column.
    ratings: a pandas Series with ratings between 1-5
    
    
    Returns:
    An object with a predict method as described in the project description
    """
  
    # The decision tree code needs the "ratings" column so that it can
    # perform tuning. Therefore, if it's not already there, we
    # will just append the Series to the end of the dataframe
    # before initializating our ratings predictor.
      
    if 'rating' not in df.columns.values:
        rp = RatingPredictor(df.join(ratings))    
    else:
        rp = RatingPredictor(df)
    rp.train()
    return rp

import os

if __name__ == '__main__':
    try:
        os.chdir('../../')
        
        # We will first need to prep the data for prediction, as per PP03
        trainDat = pd.load('proc_data/ratings_train.pda')
        uuratings = pkl.load(open('proc_data/user-user-ratings.pkl', 'rb')) # fully dimensional ratings
        iiratings = pkl.load(open('proc_data/item-item-ratings.pkl', 'rb'))
        #cfUUObj = pkl.load(open('proc_data/cf_user_object.pda', 'rb'))
        #cfIIObj = pkl.load(open('proc_data/cf_user_object.pda' ,'rb'))
        trainDat.pop('userid')
        trainDat.pop('itemid')
        trainDat.pop('isgood')
        trainDat['cf_user_rating'] = pd.cut(uuratings,np.arange(-1,6,.25))
        trainDat['cf_item_rating'] = pd.cut(iiratings,np.arange(-1,6,.25))
        
        print "Loaded data..."
    
        trueLabs = trainDat['rating'].values # Need to re-pre-process the data... so that it is prediction ready but also has ratings.
        classifier = rating_class(trainDat, trainDat['rating'])
        
        print "Trained ratings predictor, dumping him to disk..."
        fp = open('proc_data/ratingPredictor.pkl', 'wb') # for future acces
        pkl.dump(classifier, fp)
        fp.close()
        
        print "Getting test data and running predictions..."
        
        testDat = pkl.load(open('proc_data/prediction_ready_testDat.pkl', 'rb'))
        predictions = classifier.predict(testDat)
        if len(predictions) != len(trueLabs):
            raise Exception, "Predicted labels should be exactly as many as the true labels."
        meanSqErr = np.sum(np.square(predictions - trueLabs)) / len(trueLabs)
        print "Mean squared error computed: " + str(meanSqErr)
        
    except Exception as exc:
        print "An exception occurred:" + str(exc)