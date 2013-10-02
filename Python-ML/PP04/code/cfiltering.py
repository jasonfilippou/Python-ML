from __future__ import division
"""
Utilities for collaborative filtering
in a low-dimensional space
"""
import numpy as np
import pandas as pd
import os
import pickle as pkl
from numpy.linalg import norm
from PCA import pca

def make_ratings_hash(ratings):
    
    """
    Make a hashtable of ratings indexed by (userid,itemid)
    @param ratings: pandas.DataFrame of ratings
    @return a hashed version of the input ratings, which maps (userid, itemid) tuples
        to the relevant rating.
    """
    rhash = {}
    
    # for every 3-column row in the 3-column sub-dataset ratings[['userid','itemid','rating']],
    # use the values of the two first columns as the key and the third column value as the value.
    
    for _row_indx, (userid, itemid, rating) in ratings[['userid','itemid','rating']].iterrows():
        rhash[(userid,itemid)]=rating
    return rhash

def get_user_neighborhood(userVecs, curruserid, size, norms):
    
    """
    Find the nearest users and their cosine similarity to a given user
    @param userVecs -- a 2D numpy array holding the vectoral representation of users
    @param curruserid -- the id of the user whose neighborhood is being calculated
    @param size -- the number of users to be considered for user "userid"'s neighborhood
    @param norms -- a named vector (pandas.Series) of user l2 norms in rating space
    @return users -- a vector of the ids of the nearest users (the neighbors)
    @return weights -- a vector of cosine similarities for the neighbors
    """
    hash = {}
    for otheruserid in range(userVecs.shape[0]):
        if otheruserid == curruserid:
            continue
        # cosine similarity calculation
        hash[otheruserid] = userVecs[otheruserid].dot(userVecs[curruserid]) / float(norms[otheruserid] * norms[curruserid])
    # end for
    
    indx = np.argsort(-np.array(hash.values()))[:size]                  # find the indices that sort the hash by cosine similarity in DESCENDING (-) order
    users = np.array(hash.keys())[indx]                                 # and retrieve the top "size" ones. Then, retrieve both the users (keys)
    weights = np.array(hash.values())[indx]                             # and the similarities themselves
    return users, weights                                              

def make_neighborhood_hash(userids, userVecs, size, norms):
    
    """ 
    Creates the neighborhood of every user in the database.
    @param userids a numpy array holding unique user ids.
    @param userVecs a numpy array holding k-dimensional embeddings of users.
    @param size the size of the neighborhood to build for every user.
    @param norms a hash which maps users to their norms in the k-dimensional space.
    @return neighbors, a hash which maps userids to their neighbor userids
    @return weights, a hash which holds the cosine similarity between every userid pair.
    """
    
    # the following hashes will map from user_id to stuff
    
    neighbors = {}
    weights = {}

    for userid in userids:
        if userid not in neighbors:     # I believe this check is redundant now, but let's keep it anyway 
            res = get_user_neighborhood(userVecs, userid, size, norms)
            neighbors[userid], weights[userid] = res
    return neighbors, weights                   # The neighbors of every user, along with their associated cosine similarity to the user, are returned.

class CFilter(object):
    
    """
    A class to get ratings from collaborative filtering. Builds upon the provided code for PP03.
    Only difference is that it takes the user representation in terms of the low-dimensional vectors,
    instead of computing it in terms of ratings. 
    """

    def __init__(self, ratings, userVecs, size=20):
        """
        Constructor.
        @param ratings -- a pandas.DataFrame of movie ratings 
        @param userVecs: a k-space embedding of users
        @param size -- user neighborhood size (default=20)
        """
        
        self.size = size
        self.ratings = make_ratings_hash(ratings)       # ratings is now a hash with (user_id, movie_id) as the keys and movie_rating as the values
        
        # norms are computed in terms of the vectoral representation of users
        norms = {user: norm(userVecs[user]) for user in range(userVecs.shape[0])}
        self.neighbors, self.weights = make_neighborhood_hash(range(userVecs.shape[0]), userVecs, size, norms)  # notice that "self.ratings" is passed, not "ratings"

    def __repr__(self):
        return 'CFilter which implements user-user collaborative filtering, with %d ratings for %d users' % (len(self.ratings), len(self.neighbors))

    def get_user_cf_rating(self, ratings):
        """
        Get item ratings from user neighborhood

        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)

        Returns:
        A numpy array of collaborative filter item ratings from user neighborhoods. If userid is not in
        database, 0 is returned as the item rating. Ratings are discretized from 0-5 in 0.25 increments

        """
        nratings = ratings.shape[0]
        cf_rating=np.zeros(nratings)                                        # As the description mentions: "If userid is not in database..."
        for userid in self.neighbors.keys():
            indx = ratings['userid']==userid                                # indx holds indices of the ratings of user "userid", presumably. 
            if np.sum(indx)==0:                                             # ? maybe this means "no ratings found for this user".
                continue

            items = ratings['itemid'][indx].values                          # now we get the items (movies) rated by the user.
            m = len(items)                                                  # m: number of movies rated by user "userid"
            n = len(self.neighbors[userid])                                 # n: number of neighbors of user "userid"

            nratings=np.zeros((m,n))                                        # m x n ndarray holding the ratings of m users on n movies initialized to zero.
            w = np.zeros((m,n))                                             # same, but this time this array holds weights (cosine similarities)

            for i in xrange(m):                                             # for every item (movie)
                itemid = items[i]                                           # get its id
                for j in xrange(n):                                         # for every neighbor of the user "userid"
                    ouid = self.neighbors[userid][j]                        # get his id
                    if (ouid, itemid) in self.ratings:                      # if the (neighbor_id, movie_id) pair exists in the ratings map,
                        nratings[i,j] = self.ratings[(ouid,itemid)]         # store the rating of the other user
                        w[i,j] = self.weights[userid][j]                    # and the weight (cosine similarity between the current user and the neighbor)
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

if __name__ == "__main__":
    
    os.chdir("..")
    ratings_train = pd.load('proc_data/ratings_train.pda')
   
    accuracyList = list() # a list of (embedding, deviation_from_rating, misclassification_error) tuples
    for k in range(2, 1677, 20):
        
        #Get user embedding
        
        userEmbedding = pca(ratings_train[['userid', 'itemid', 'rating']], k, 0).real
        print "Computed %d-dimensional user embedding." %(k)
        
        # Calculate collaborative filtering ratings
        
        cf = CFilter(ratings_train, userEmbedding, size = 20)
        print "Built CFilter object"
        predictedRatings = cf.get_user_cf_rating(ratings_train)
        
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
        
    #pkl.dump(accuracyList, open('output_data/accuracyList.pkl', 'wb'))
    print "Done. Exiting..."
    
    #### return sorted(featureList, key=lambda data: data[2])[::-1]
    