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

from mlExceptions import LogicalError, DatasetError

def make_ratings_hash(ratings):
    """Make a hashtable of ratings indexed by (userid,itemid)"""
    rhash = {}
    
    # for every 3-column row in the 3-column sub-dataset ratings[['userid','itemid','rating']],
    # use the values of the two first columns as the key and the third column value as the value.
    
    for _row_indx, (userid, itemid, rating) in ratings[['userid','itemid','rating']].iterrows():
        rhash[(userid,itemid)]=rating
    return rhash

def get_item_neighborhood(ratings, itemid, size, norms):
    """
    Find the nearest items and their cosine similarity to a given item

    Arguments:
    ratings -- a ratings hash table as produced by make_ratings_hash
    itemid -- the id of the item whose neighborhood is being calculated
    size -- the number of items to be considered for item "itemid"'s neighborhood
    norms -- a named vector (pandas.Series) of user l2 norms in rating space

    Returns:
    users -- a vector of the ids of the nearest users (the neighbors)
    weights -- a vector of cosine similarities for the neighbors
    """
    hash = {}
    
    for (userid,otheritemid),rating in ratings.iteritems():             # remember that ratings is a hash table
        if otheritemid == itemid:
            continue                                                    # doesn't make much sense to compute the distance to ourselves.
        if (userid, itemid) not in ratings:                             # if the current movie hasn't been rated at all
            continue                                                    # it is not suitable to compare similarity from

        if otheritemid not in hash:                                     # If you haven't stored the item currently considered
            hash[otheritemid] = 0                                       # in your hash, do it now.

        hash[otheritemid] += ratings[(userid,itemid)] * rating          # slowly building the dot product of the numerator of the cosine similarity.
    # end for
    
    for (otheritemid, _rating) in hash.iteritems():
        nx=norms[itemid]
        ny=norms[otheritemid]
        if ny == 0:
            raise LogicalError, "Norm of item id: %d detected to be zero." %(itemid)
        hash[otheritemid] = hash[otheritemid]/float(nx*ny)              # there you have it, the full cosine similarity between "userid" and "otheruserod"
    # end for
    
    indx = np.argsort(-np.array(hash.values()))[:size]                  # find the indices that sort the hash by cosine similarity in DESCENDING (-) order
    items = np.array(hash.keys())[indx]                                 # and retrieve the top 20 ones. Then, retrieve both the users (keys)
    weights = np.array(hash.values())[indx]                             # and the similarities themselves
    return items, weights                                              

def make_neighborhood_hash(itemids, ratings, size, norms):
    
    """ Creates the neighborhood of every user in the database"""
    
    # the following hashes will map from user_id to stuff
    
    neighbors = {}
    weights = {}

    for itemid in itemids:
        if itemid not in neighbors:     # put it in 
            res = get_item_neighborhood(ratings, itemid, size, norms)
            neighbors[itemid], weights[itemid] = res
    return neighbors, weights                   # The neighbors of every user, along with their associated cosine similarity to the user, are returned.

class CFilter_item(object):
    """A class to get ratings from collaborative filtering"""

    def __init__(self, ratings, size=20):
        """
        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)
        size -- item neighborhood size (default=20)
        """
        self.size = size
        self.ratings = make_ratings_hash(ratings)       # ratings is now a hash with (user_id, movie_id) as the keys and movie_rating as the values
        
        # Compute the two-norm for every item's ratings, since we need it for the cosine similarity
        norms = ratings[['itemid','rating']].groupby('itemid').aggregate(lambda x: np.sqrt(np.sum(x**2)))['rating']    
        itemids = ratings['itemid']
        self.neighbors, self.weights = make_neighborhood_hash(itemids, self.ratings, size, norms)  # notice that "self.ratings" is passed, not "ratings"

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
    
    
########## our testing code ###############

import pandas as pd
import os 
import pickle as pkl

def main():
    os.chdir("../../")
    
    # Load the DataFrame representing the MovieLens subset
    
    ratings=pd.load('input_data/ratings_train.pda')
    ratings = ratings.dropna()
    nratings = ratings.shape[0]
    
    try:
        
        itemFilter = CFilter_item(ratings) 
        item_cf_ratings = itemFilter.get_cf_rating(ratings)
        
        # Compute average deviation from rating on those ratings
        avgDeviation = np.sum(np.abs((item_cf_ratings - ratings['rating']))) / len(ratings)
        print "Average deviation from rating: %.4f" %(avgDeviation)
        
        # Dump the ratings to disk for easy future access
        fp = open("proc_data/item_item_ratings_2.pda", "wb")
        pkl.dump(item_cf_ratings, fp)
        fp.close()
                
        print "This concludes the main method. Buh-bye!"
        
    except LogicalError as l:
        print "A logical error occurred: %s" %(l)
    except DatasetError as d:
        print "A dataset-related error occurred: %s" %(d)
    except Exception as e:
        print "An exception occurred: " + str(e)
    
if __name__ == "__main__":
    main()