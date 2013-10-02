from __future__ import division
'''
Created on Nov 3, 2012

@author: Jason
'''

"""
Extension of collaborative filtering utilities
to perform item-item collaborative filtering.

This version represents each movie as the 
vector of ratings made for that movie.
"""
import numpy as np

from util.mlExceptions import LogicalError, DatasetError
from scipy.linalg import norm
from PCA import pca
from math import isnan

def make_ratings_hash(ratings):
    """Make a hashtable of ratings indexed by (userid,itemid)"""
    rhash = {}
    
    # for every 3-column row in the 3-column sub-dataset ratings[['userid','itemid','rating']],
    # use the values of the two first columns as the key and the third column value as the value.
    
    for _row_indx, (userid, itemid, rating) in ratings[['userid','itemid','rating']].iterrows():
        rhash[(userid,itemid)]=rating
    return rhash

def get_item_neighborhood(itemVecs, curritemid, size, norms):
    
    """
    Find the nearest users and their cosine similarity to a given user
    @param itemrVecs -- a 2D numpy array holding the vectoral representation of items
    @param curruserid -- the id of the user whose neighborhood is being calculated
    @param size -- the number of users to be considered for user "userid"'s neighborhood
    @param norms -- a named vector (pandas.Series) of user l2 norms in rating space
    @return users -- a vector of the ids of the nearest users (the neighbors)
    @return weights -- a vector of cosine similarities for the neighbors
    """
    
    hash = {}
    for otheritemid in range(itemVecs.shape[0]):
        if otheritemid == curritemid:
            continue
        # cosine similarity calculation
        if otheritemid in norms and curritemid in norms:
            hash[otheritemid] = itemVecs[otheritemid].dot(itemVecs[curritemid]) / float(norms[otheritemid] * norms[curritemid])
    # end for
    
    indx = np.argsort(-np.array(hash.values()))[:size]           # find the indices that sort the hash by cosine similarity in DESCENDING (-) order
    items = np.array(hash.keys())[indx]                          # and retrieve the top "size" ones. Then, retrieve both the users (keys)
    weights = np.array(hash.values())[indx]                      # and the similarities themselves
    return items, weights                                             

def make_neighborhood_hash(itemids, itemVecs, size, norms):
    
    """ Creates the neighborhood of every user in the database"""
    
    # the following hashes will map from user_id to stuff
    
    neighbors = {}
    weights = {}

    for itemid in itemids:
        if itemid not in neighbors:     # put it in 
            res = get_item_neighborhood(itemVecs, itemid, size, norms)
            neighbors[itemid], weights[itemid] = res
    return neighbors, weights                   # The neighbors of every user, along with their associated cosine similarity to the user, are returned.

class CFilter_item(object):
    """A class to get ratings from collaborative filtering"""

    def __init__(self, ratings, itemVecs, size=20):
        """
        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)
        size -- item neighborhood size (default=20)
        """
        self.size = size
        self.ratings = make_ratings_hash(ratings)       # ratings is now a hash with (user_id, movie_id) as the keys and movie_rating as the values
        
        norms = {item: norm(itemVecs[item]) for item in range(itemVecs.shape[0]) if not isnan(itemVecs[item][0])}    
        self.neighbors, self.weights = make_neighborhood_hash(range(itemVecs.shape[0]), itemVecs, size, norms)  # notice that "self.ratings" is passed, not "ratings"

    def __repr__(self):
        return 'CFilter_item which implements item-item collaborative filtering, with %d ratings for %d items' % (len(self.ratings), len(self.neighbors))

    def get_item_cf_rating(self, ratings):
        """
        Get item ratings from item neighborhood

        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)

        Returns:
        A numpy array of collaborative filter item ratings from user neighborhoods. If userid is not in
        database, 0 is returned as the item rating. Ratings are discretized from 0-5 in 0.25 increments

        """
        nratings = ratings.shape[0]
        cf_rating=np.zeros(nratings)                                        # As the description mentions: "If userid is not in database..."

        for itemid in self.neighbors.keys():
            indx = ratings['itemid']==itemid                                # indx holds indices of the ratings of user "userid", presumably. 
            if np.sum(indx)==0:                                             
                continue

            users_who_rated = ratings['userid'][indx].values                # now we get the users who rated this movie.
            m = len(users_who_rated)                                        # m: number of users who rated the movie "itemid"
            n = len(self.neighbors[itemid])                                 # n: number of neighbors of item "itemid"

            nratings=np.zeros((m,n))                                        # m x n ndarray holding the ratings of m users on n movies initialized to zero.
            w = np.zeros((m,n))                                             # same, but this time this array holds weights (cosine similarities)

            for i in xrange(m):                                             # for every user who rated
                current_user_id = users_who_rated[i]                        # get his id
                for j in xrange(n):                                         # for every neighbor of the item "itemid"
                    otheritemid = self.neighbors[itemid][j]                 # get his id
                    if (current_user_id, otheritemid) in self.ratings:      # if the currently examined rater has rated the neighbor
                        nratings[i,j] = self.ratings[(current_user_id,otheritemid)]         # store the rating
                        w[i,j] = self.weights[itemid][j]                    # and the weight (cosine similarity between the current item and the neighbor item
                    #end if
                #end for
            #end for
              
            sw = np.sum(w,axis=1)                                           # sw (sum of w): 1D array that maintains the sum of the cosine similarities of every user
            keep = sw>0                                                     # keep: an array with boolean True/False values. True if the same position at sw is > 0, False otherwise.
            if np.sum(keep)==0:                                             # Another way of checking if the "keep" nd array is empty. If it is,
                continue                                                    # no neighbor rated the movie above zero. In this case, continue on to the next userid.

            nratings *= w                                                   # Multiply the neighbors' ratings by the neighbors' cosine similarities.
            res = np.sum(nratings,axis=1)                                   # Retrieve the sum of the rating of every movie (which has been multiplied by the respective neighbor's cosine similarity)
                                                                            # and store it in a 1D array

            res[keep.nonzero()] /= sw[keep.nonzero()]                       # Normalize by the sum of cosine similarities of all neighbors.
            cf_rating[indx] = res                                           # For every item rated by the user "user_id", we now also have the collaborative filtering rating of the neighborhood.
            
        # end for user_id
        
        return cf_rating

import os
import pandas as pd
import pickle as pkl

if __name__ == "__main__":
    
    os.chdir("..")
    ratings_train = pd.load('proc_data/ratings_train.pda')
   
    accuracyList = list() # a list of (embedding, deviation_from_rating, misclassification_error) tuples
    for k in range(2, 941, 20):
        
        #Get user embedding
        
        itemEmbedding = pca(ratings_train[['userid', 'itemid', 'rating']], k, 1).real
        #print itemEmbedding
        print "Computed %d-dimensional user embedding." %(k)
        
        # Calculate collaborative filtering ratings
        
        cf = CFilter_item(ratings_train, itemEmbedding, size = 20)
        print "Built CFilter object"
        predictedRatings = cf.get_item_cf_rating(ratings_train)
        
        # Estimate training error in terms of two metrics: average deviation from true
        # rating and average misclassification error in terms of classifying a movie
        # as "good" or "bad".
        
        predictedLabels = np.array([1 if rat > 3 else 0 for rat in predictedRatings])
        # make the "isgood" column of the data map to {0,1} instead of {-1, 1}
        # so that average squared loss works correctly
        trueLabels = np.array([1 if lab == 1 else 0 for lab in ratings_train['isgood']])
        avgRatingDeviation = np.sum(np.abs(predictedRatings- ratings_train['rating'])) / float(ratings_train.shape[0])
        avgSqLoss = np.sum(np.square(predictedLabels - trueLabels)) / float(ratings_train.shape[0])
        accuracyList.append((k, avgRatingDeviation, avgSqLoss))
        print "Predicted rating for embedding %d. Average deviation from true rating: %.4f. Average squared loss: %.4f%%." %(k, avgRatingDeviation,100*avgSqLoss)
        
    pkl.dump(accuracyList, open('output_data/accuracyList_item_item.pkl', 'wb'))
    print "Done. Exiting..."