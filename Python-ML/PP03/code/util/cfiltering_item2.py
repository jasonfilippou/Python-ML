from __future__ import division
'''
Created on Nov 3, 2012

@author: Jason
'''

"""
Extension of collaborative filtering utilities
to perform item-item collaborative filtering

This version represents each movie by a vector
consisting of all its possible genres, as well
as the decade in which it was released.
"""

import numpy as np
from mlExceptions import LogicalError, DatasetError


def make_ratings_hash(ratings):
    
    """
    Make a hashtable of ratings indexed by itemId and pointing to
    the vector (genres, decade) that fully characterize an item.
    """
    
    rhash = {}
     
    # For every rating, check if the relevant item is already in the map.
    # If not, add it to the map. Key is item_id, mapped value is the vector
    # consisting of the possible genres and the decade of the movie.
    
    for row_indx, itemid in ratings['itemid'].iteritems():
        if itemid not in rhash:
            itemData = ratings.ix[row_indx, 'Action' : 'decade']
            rhash[itemid] = itemData
    return rhash

def compute_norms(items):
    
    """
    Compute the norms of the item vectors provided.
    
    Arguments:
    items -- a hashmap which maps itemIDs to the characteristic vectors
    """
    
    norms = {}
    
    for item in items:
        norms[item] = np.sqrt(np.sum(np.square(items[item])))
        
    return norms
    
def get_item_neighborhood(ratings, itemid, size, norms):
    
    """
    Find the nearest items and their cosine similarity to a given item

    Arguments:
    ratings -- a hash with itemid keys and item vector values.
    item -- the id of the item whose neighborhood is being calculated
    size -- the number of items to be considered for item "itemid"'s neighborhood
    norms -- a named vector (pandas.Series) of item l2 norms in rating space

    Returns:
    items -- a vector of the ids of the nearest items (the neighbors)
    weights -- a vector of cosine similarities for the neighbors
    """
    
    # Create a hash which will have the current itemid as the key
    # and will map to the dot products of the current item's vectorized interpretation
    # and all the different neighbor items.
    
    similarityhash = {}
    
    for otheritemid, otheritemvector in ratings.iteritems():                        # remember that ratings is a hash table
        if otheritemid == itemid:
            continue                                                                # doesn't make much sense to compute the distance to ourselves.

        if otheritemid not in similarityhash:                                       # If you haven't stored the item currently considered
            similarityhash[otheritemid] = 0                                         # in your hash, do it now.

        if len(ratings[itemid]) != len(otheritemvector):
            raise LogicalError ,"Cannot compute the dot product of mismatch size vectors"
        similarityhash[otheritemid] = np.dot(ratings[itemid], otheritemvector)
    
    
    # Now we will loop through the hash and update each value 
    # by dividing it with the product of the current item's norm
    # and the neighbor's norm. This completes computation of the cosine
    # similarities between the current movie and all other movies in
    # the database.
    
    for otheritemid, _otheritemvector in similarityhash.iteritems():
        nx=norms[itemid]
        ny=norms[otheritemid]
        similarityhash[otheritemid] = similarityhash[otheritemid]/float(nx*ny)   # there you have it, the full cosine similarity between "itemid" and "otheritemd"
    
    
    # Now that we have the cosine similarities between the current
    # movie and all other movies, we will create the current movie's
    # neighborhood by considering only the "size" most similar movies
    # to the current movie (as dictated by the cosine similarities, i.e
    # the values of the hash).
    
    indx = np.argsort(-np.array(similarityhash.values()))[:size]                  # find the indices that sort the hash by cosine similarity in DESCENDING (-) order
    
    # Finally, we will return both the itemIDs of the neighborhood,
    # as well as the cosine similarities of the neighborhood.
    
    items = np.array(similarityhash.keys())[indx]                                 # and retrieve the top 20 ones. Then, retrieve both the items (keys)
    weights = np.array(similarityhash.values())[indx]                             # and the similarities themselves
    return items, weights                                              

def make_neighborhood_hash(itemids, ratings, size, norms):
    
    """ 
    Creates the neighborhood of every user in the database
    
    Arguments:
    
    itemids -- pandas.Series object which holds all the different item ids in the database
    ratings -- a hash with item id keys and item vector values.
    size -- an integer which dictates the maximum number of neighbors to consider.
    norms -- a hash with item id keys and float values representing the norms of the respective items.
    """
    
    # the following hashes will map from item_id to stuff
    
    neighbors = {}
    weights = {}

    for itemid in itemids:
        if itemid not in neighbors:                                         # put it in 
            res = get_item_neighborhood(ratings, itemid, size, norms)
            neighbors[itemid], weights[itemid] = res                        # now we have the neighbor items' ids and their cosine similarities to the current item id, associated with the current item id
            
            # Some sanity checks
            
            if len(neighbors[itemid]) != size:
                raise LogicalError ,"Total number of neighbors of item id %s should be %d." %(itemid, size)
            if len(weights[itemid]) != len(neighbors[itemid]):
                raise LogicalError ,"The number of cosine similarities considered for item id %s should be the same as its neighbors." %(itemid)
            
    return neighbors, weights                                               # The neighbors of every item, along with their associated cosine similarity to the item, are returned.

def extractRating(rater_id, movie_id, ratings):
    
    """
    A method that extracts the rating at one position of the data. 
    If the rating is non-existent, it returns zero.
    
    Arguments: 
    rater_id -- integer representing the rater id
    movie_id -- integer represeting the movie id. Along with rater id, it uniquely characterizes an enty
    ratings -- pandas.Dataframe which holds our ratings data.
    
    Returns:
    The relevant rating, or 0 if it doesn't exist. 
    """
    
    # Sanity checking first
    
    if len(ratings[ratings['userid'] == rater_id]) == 0:
        raise LogicalError ,"No ratings detected for user: %s." %(rater_id)
    
    
    # Find all ratings of the user provided, and then constrain those
    # to the single rating corresponding to the (rater_id, movie_id) pair.
    ratingsOfUser = ratings[ratings['userid'] == rater_id]
    specificRating = ratingsOfUser[ratingsOfUser['itemid'] == movie_id]['rating']
    
    # Return 0 if there exists no rating, or actual rating otherwise
    
    return 0 if specificRating.values.size == 0 else specificRating.values[0]

class CFilter_item2(object):
    
    """
    A class to get ratings from item-item collaborative filtering
    """

    def __init__(self, ratings, size=20):
        
        """
        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)
        size -- item neighborhood size (default=20)
        """
        
        self.size = size
        self.itemHash = make_ratings_hash(ratings)      # self.itemHash is now a hash with item_id as the keys 
                                                        # and [Action, ...., Western, decade] as the values
        
        # Compute the two-norm for every item's representative vector, 
        # since we need it for the cosine similarity
        norms = compute_norms(self.itemHash)                  # norms is now a hash that maps item_id to norm of item_id
        itemids = ratings['itemid']                         # note that this is not a unique Series.
        self.neighbors, self.weights = make_neighborhood_hash(itemids, self.itemHash, size, norms)  

    def __repr__(self):
        return 'CFilter_item which implements item-item collaborative filtering, with %d ratings for %d items' % (len(self.ratings), len(self.neighbors))

    def get_item_2_cf_rating(self, ratings):
        
        """
        Get item ratings from item neighborhood

        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)

        Returns:
        A numpy array of collaborative filter item ratings from item neighborhoods. If itemid is not in
        database, 0 is returned as the item rating. Ratings are discretized from 0-5 in 0.25 increments.

        """
        nratings = ratings.shape[0]
        if nratings == 0:
            raise DatasetError ,"Empty dataset provided."
        cf_rating=np.zeros(nratings)                                        # As the description mentions: "If itemid is not in database..." 
        for itemid in self.neighbors.keys():                                # For every item id in the database
            indx = ratings['itemid']==itemid                                # indx holds indices of the ratings of the current item examined. 
            if np.sum(indx)==0:                                             # No ratings found for this movie. Continue on to the next movie.
                continue                                                    # this is consistent with the fact that we want a movie without ratings to be collaboratively rated as "zero"

            users_who_rated = ratings['userid'][indx].values                # store the ids of the users who rated the current item
            m = np.sum(indx)                                                # m: number of ratings of this item.
            n = len(self.neighbors[itemid])                                 # n: number of neighbors of this item.

            nratings=np.zeros((m,n))                                        # m x n ndarray holding the collaborative ratings of the current item.
            w = np.zeros((m,n))                                             # same, but this time this array holds weights (cosine similarities)

            for i in xrange(m):                                             # for every rating of the currently examined movie
                currentRater = users_who_rated[i]                           # fetch the rater
                for j in xrange(n):                                         # for every neighbor item of the item "itemid"
                    otheritemid = self.neighbors[itemid][j]                 # get its id
                    nratings[i, j] = extractRating(currentRater, 
                                                   otheritemid, ratings)    # store the specific rating corresponding to (currentRater, otheritemid) (or 0 if it doesn't exist)
                    # Sanity check: is rating within bounds?
                    if nratings[i, j] < 0:
                        raise LogicalError, " for userid %d and movieid %d, extractRating returned a negative rating." %(currentRater, otheritemid)
                    if nratings[i, j] > 5:
                        raise LogicalError, " for userid %d and movieid %d, extractRating returned a rating above 5." %(currentRater, otheritemid) 
                    w[i,j] = self.weights[itemid][j]                        # and the weight (cosine similarity between the current user and the neighbor)
                #end for
            #end for
              
            sw = np.sum(w,axis=1)                                           # sw (sum of w): 1D array that maintains the SUM of the consine similarities of the current item and every item in its neighborhood
            
            # The following three lines of code are not useful for us 
            # in this item-item collaborative filtering modification 
            # of the original code, because the presence of the movie release
            # decade in the vector representing each movie ensures that the dot
            # product will always be > 0, so there will be no non-zero sums. 
            # However, out of pure fear that maybe we haven't thought something through, 
            # we will keep all references to the vector "keep". No pun intended,
            # but nobody cares whether our pun was intended anyway.
            
            keep = sw>0                                                     
            if np.sum(keep)==0:                                             
                continue                                                    

            nratings *= w                                                   # Multiply the neighbors' ratings by the neighbors' cosine similarities. This multiplies the matrices element-wise.
            res = np.sum(nratings,axis=1)                                   # For every rating of the current movie, retrieve the sum of the weighted ratings of the neighbor movies.

            res[keep.nonzero()] /= sw[keep.nonzero()]                       # Normalize by the sum of cosine similarities of all neighbors.
            cf_rating[indx] = res                                           # For every item rated, we now also have the collaborative filtering rating of the neighborhood.
            
        # end for item_id
        
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
    
    # Load the CFiltering_item object 
    
    fp = open("proc_data/cfilter_object.pda", "rb")
    cf3 = pkl.load(fp)
    fp.close()
    
    try:
        
        # we need to make sure everything's ok with our code
        # so we will do some tests on dimished MovieLens data
        
        item_cf_ratings = cf3.get_cf_rating(ratings.ix[:10000,:])
    
        # Compute squared loss on those ratings
        
        avgSquaredLoss = np.sum(np.square(item_cf_ratings - ratings.ix[:10000, :]['rating'])) / len(ratings)
        print avgSquaredLoss        # 1.59836284936 on the first 10000 items!
 
        
    except LogicalError as l:
        print "A logical error occurred: %s" %(l)
    except DatasetError as d:
        print "A dataset-related error occurred: %s" %(d)
    except Exception as e:
        print "An exception occurred: " + str(e)
    
if __name__ == "__main__":
    main()