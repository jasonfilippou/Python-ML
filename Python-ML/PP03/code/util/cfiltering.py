"""
Utilities for collaborative filtering
"""
import numpy as np
import pandas as pd

def make_ratings_hash(ratings):
    """Make a hashtable of ratings indexed by (userid,itemid)"""
    rhash = {}
    
    # for every 3-column row in the 3-column sub-dataset ratings[['userid','itemid','rating']],
    # use the values of the two first columns as the key and the third column value as the value.
    
    for _row_indx, (userid, itemid, rating) in ratings[['userid','itemid','rating']].iterrows():
        rhash[(userid,itemid)]=rating
    return rhash

def get_user_neighborhood(ratings, userid, size, norms):
    """
    Find the nearest users and their cosine similarity to a given user

    Arguments:
    ratings -- a ratings hash table as produced by make_ratings_hash
    userid -- the id of the user whose neighborhood is being calculated
    size -- the number of users to be considered for user "userid"'s neighborhood
    norms -- a named vector (pandas.Series) of user l2 norms in rating space

    Returns:
    users -- a vector of the ids of the nearest users (the neighbors)
    weights -- a vector of cosine similarities for the neighbors
    """
    hash = {}
    for (otheruserid,itemid),rating in ratings.iteritems():             # remember that ratings is a hash table
        if otheruserid == userid:
            continue                                                    # doesn't make much sense to compute the distance to ourselves.
        if (userid, itemid) not in ratings:                             # if the current movie hasn't been rated by us at all
            continue                                                    # it is not suitable to compare similarity from

        if otheruserid not in hash:                                     # If you haven't stored the user currently considered
            hash[otheruserid] = 0                                       # in your hash, do it now.

        hash[otheruserid] += ratings[(userid,itemid)] * rating          # slowly building the dot product of the numerator of the cosine similarity.
    # end for
    
    for (otheruserid, _val) in hash.iteritems():
        nx=norms[userid]
        ny=norms[otheruserid]
        hash[otheruserid] = hash[otheruserid]/float(nx*ny)              # there you have it, the full cosine similarity between "userid" and "otheruserod"
    # end for
    
    indx = np.argsort(-np.array(hash.values()))[:size]                  # find the indices that sort the hash by cosine similarity in DESCENDING (-) order
    users = np.array(hash.keys())[indx]                                 # and retrieve the top 20 ones. Then, retrieve both the users (keys)
    weights = np.array(hash.values())[indx]                             # and the similarities themselves
    return users, weights                                              

def make_neighborhood_hash(userids, ratings, size, norms):
    
    """ Creates the neighborhood of every user in the database"""
    
    # the following hashes will map from user_id to stuff
    
    neighbors = {}
    weights = {}

    for userid in userids:
        if userid not in neighbors:     # put him in 
            res = get_user_neighborhood(ratings, userid, size, norms)
            neighbors[userid], weights[userid] = res
    return neighbors, weights                   # The neighbors of every user, along with their associated cosine similarity to the user, are returned.

class CFilter(object):
    """A class to get ratings from collaborative filtering"""

    def __init__(self, ratings, size=20):
        """
        Arguments:
        ratings -- a pandas.DataFrame of movie ratings (see prep_data)
        size -- user neighborhood size (default=20)
        """
        self.size = size
        self.ratings = make_ratings_hash(ratings)       # ratings is now a hash with (user_id, movie_id) as the keys and movie_rating as the values
        
        # Compute the two-norm for every user's ratings, since we need it for the cosine similarity
        norms = ratings[['userid','rating']].groupby('userid').aggregate(lambda x: np.sqrt(np.sum(x**2)))['rating']    
        userids = ratings['userid']
        self.neighbors, self.weights = make_neighborhood_hash(userids, self.ratings, size, norms)  # notice that "self.ratings" is passed, not "ratings"

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