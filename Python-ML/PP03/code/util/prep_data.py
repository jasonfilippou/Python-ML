import numpy as np
import pandas as pd
from datetime import datetime as dt

def get_decade(dateSeries):
    """Turn a pandas.Series of dates to a Series of integers indicating the decade."""

    tmp = zip(dateSeries, pd.isnull(dateSeries))
    f = lambda datestr: int(dt.strptime(datestr, '%d-%b-%Y').strftime('%y'))
    g = lambda year: year - (year % 10)
    return [g(f(datestr)) if not missing else None for (datestr,missing) in tmp]

def prep_data(datadir, ratings_file, numratings=10000):
    """Make a pandas.DataFrame of movie ratings from the movie lens database

    Arguments:
    datadir -- the directory where the movie lens database is located
    ratings_file -- the file to extract ratings from
    numratings -- how many ratings to include in table (default=10,000)

    Returns:
    pandas.DataFrame with columns:
      userid -- the user's id
      itemid -- the item's id
      rating -- the user's movie rating (integer, 1-5)
      age -- the user's age (discretized from 0-100 in steps of 5)
      gender -- the user's gender ("M" or "F")
      occupation -- the user's occupation
      Action -- binary (0,1) genre indicator
      Adventure -- binary (0,1) genre indicator
      Animation -- binary (0,1) genre indicator
      Children's -- binary (0,1) genre indicator
      Comedy -- binary (0,1) genre indicator
      Crime -- binary (0,1) genre indicator
      Documentary -- binary (0,1) genre indicator
      Drama -- binary (0,1) genre indicator
      Fantasy -- binary (0,1) genre indicator
      Film-Noir -- binary (0,1) genre indicator
      Horror -- binary (0,1) genre indicator
      Musical -- binary (0,1) genre indicator
      Mystery -- binary (0,1) genre indicator
      Romance -- binary (0,1) genre indicator
      Sci-Fi -- binary (0,1) genre indicator
      Thriller -- binary (0,1) genre indicator
      War -- binary (0,1) genre indicator
      Western -- binary (0,1) genre indicator
      decade -- integer, the movie's release decade
      isgood -- binary (+1,-1) indicator if rating is greater than 3

    """

    ratings_file = '%s/%s' % (datadir, ratings_file) # some sort of concatenation, apparently

    # read ratings data
    ratings = pd.read_table(ratings_file, delimiter='\t', header=None)
    ratings.columns = ['userid','itemid','rating','timestamp']

    # sample numratings ratings from the table
    nratings = ratings.shape[0]
    
    # if the number of ratings provided as input to this function
    # is less than the total number of ratings in the data,
    # we will take a random sample of the data and consider
    # that to be our training data.
    
    if numratings is not None and numratings < nratings:
        
        # shuffle the order of all the ratings and retrieve the first
        # "numratings" ones
        
        _sample=np.random.permutation(np.arange(nratings))[:numratings-1]   
        ratings = ratings.ix[:numratings,:]     # now "ratings" is a reduced pandas.DataFrame

    # drop the timestamp column
    ratings.pop('timestamp')

    # read user data
    users_file = '%s/u.user' % datadir
    users = pd.read_table(users_file, delimiter="|", header=None)
    users.columns = ['userid','age','gender','occupation','zipcode']

    # drop the zipcode column
    users.pop('zipcode')

    # discretize the age data
    users['age']=pd.cut(users['age'],np.arange(0,100,5))    

    # add user info to ratings data
    ratings = pd.merge(ratings, users)  # expands the "ratings" pandas.DataFrame

    # get movie data
    items_file = '%s/u.item' % datadir
    items = pd.read_table(items_file, delimiter='|', header=None)

    # get genre information
    genre_file = '%s/u.genre' % datadir
    genres = pd.read_table(genre_file, delimiter='|', header=None)
    genres = list(genres.ix[:genres.shape[0]-2,0].values)

    items.columns = ['itemid','title','releasedate','videodate','url'] + genres

    # get the movie decade
    items['decade']=get_decade(items['releasedate'])

    # drop columns
    for col in ['title', 'releasedate',' videodate', 'url']:
        if col in items:
            items.pop(col)

    ratings = pd.merge(ratings, items)  # Again expanding our data

    # remove user and item ids
    #ratings.pop('userid')
    #ratings.pop('itemid')

    # binarize ratings
    ratings['isgood'] = [1 if rating > 3 else -1 for rating in ratings['rating']]

    # fix indexing
    ratings.index = np.arange(ratings.shape[0])
    return ratings

