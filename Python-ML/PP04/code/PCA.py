'''
Created on Nov 29, 2012

@author: jason
'''
from inspect import stack
import pandas as pd
import numpy as np
import os
from util.mlExceptions import DatasetError, LogicalError
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs
from numpy.linalg import eig
# Global constant for current function name

CURR_FUNC_NAME = stack()[0][3]

def pca(df, k, axis = 0):
    
    """
    This PCA implementation is geared towards solving question 1 of the programming assignment.
    Effectively, the df array is constrained to contain only three columns: userid, movieid and 
    rating. We will then create a sparse |unique_user_id| X |unique_movie_id| matrix (or the other
    way round, dependent on the value of "axis", which we will then embed in k dimensions. 
    
    @param df: a two-dimensional pandas DataFrame with columns userid, itemid and rating
    @param axis: which axis to treat as examples (0 or 1)
    @param k: number of dimensions in embedding
    @return numpy matrix of shape (m,k) where m is the number of unique userids in df when axis=0 
        and m is the number of unique itemids in df when axis=1
    @raise DatasetError when the dataset provided is None or empty
    @raise LogicalError when axis is neither 0 nor 1, or k <= 0
    """
    
    # Sanity checking
    
    if axis not in [0, 1]:
        raise LogicalError, "Method %s: \"axis\" variable should be either 0 or 1 (provided: %s)." %(CURR_FUNC_NAME, str(axis))
    if k <= 0 or not isinstance(k, int):
        raise LogicalError, "Method %s: number k of embedding dimensions should be a positive integer (provided: %s)." %(CURR_FUNC_NAME, str(k))
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise DatasetError, "Method %s: empty dataset provided." %(CURR_FUNC_NAME)
    
    if len(df.columns.values) != 3:
        raise DatasetError, "Method %s: the dataframe provided should have exactly 3 columns." %(CURR_FUNC_NAME)
    
    if 'userid' not in df.columns.values[0] or 'itemid' not in df.columns.values or 'rating' not in df.columns.values:
        raise DatasetError, "Method %s: the dataframe provided should have 3 columns named \"userid\", \"itemid\" and \"rating\"." %(CURR_FUNC_NAME)
    
    # Load the dataframe values in a hash. It will make life easier.
    
    ratingsHash = {}
    for _row_indx, (userid, itemid, rating) in df.iterrows():
        ratingsHash[(userid, itemid)] = rating
    
    # We now need to make our m x n sparse array.
    
    rowIndex = 'userid' if axis == 0 else 'itemid'
    columnIndex = 'itemid' if axis == 0 else 'userid'
    uniqueRows = df[rowIndex].unique()
    uniqueCols = df[columnIndex].unique()
    
    sparseArr = np.zeros((len(uniqueRows), len(uniqueCols)))    # zerofill initially
    for i in range(len(uniqueRows)):
        for j in range(len(uniqueCols)):
            if (uniqueRows[i], uniqueCols[j]) in ratingsHash:
                sparseArr[i][j] = ratingsHash[(uniqueRows[i], uniqueCols[j])]

    # Compute the covariance matrix
    print "sparseArr shape: " + str(sparseArr.shape)
    covMat = np.cov(sparseArr.T) 
    
    # A compressed representation is needed because we need to center sparse data.
     
    csr_rep = lil_matrix(sparseArr).tocsr() 
    for c in range(csr_rep.shape[axis]):
        sparseArr[c, :][sparseArr[c, :].nonzero()] -= np.mean(csr_rep.getcol(c).data)
    
    # Find eigenvalues, compute and return k-dimensional embedding
    
    print "covMat shape: " + str(covMat.shape)
    eigenVals, eigenVecs = eigs(covMat, k)
    
    # Re-arrange eigenvectors so that you get the most significant components first.
    eigenVecs = eigenVecs[:, np.argsort(-eigenVals)]
    return sparseArr.dot(eigenVecs) 

    
def pca2(df, k, axis = 0):
    
    """
    This PCA implementation is geared towards solving question 1 of the programming assignment.
    Effectively, the df array is constrained to contain only three columns: userid, movieid and 
    rating. We will then create a sparse |unique_user_id| X |unique_movie_id| matrix (or the other
    way round, dependent on the value of "axis", which we will then embed in k dimensions. 
    
    @param df: a two-dimensional pandas DataFrame with columns userid, itemid and rating
    @param axis: which axis to treat as examples (0 or 1)
    @param k: number of dimensions in embedding
    @return numpy matrix of shape (m,k) where m is the number of unique userids in df when axis=0 
        and m is the number of unique itemids in df when axis=1
    @raise DatasetError when the dataset provided is None or empty
    @raise LogicalError when axis is neither 0 nor 1, or k <= 0
    """
    
    # Sanity checking
    
    if axis not in [0, 1]:
        raise LogicalError, "Method %s: \"axis\" variable should be either 0 or 1 (provided: %s)." %(CURR_FUNC_NAME, str(axis))
    if k <= 0 or not isinstance(k, int):
        raise LogicalError, "Method %s: number k of embedding dimensions should be a positive integer (provided: %s)." %(CURR_FUNC_NAME, str(k))
    if df is None or df.shape[0] == 0 or df.shape[1] == 0:
        raise DatasetError, "Method %s: empty dataset provided." %(CURR_FUNC_NAME)
    
    if len(df.columns.values) != 3:
        raise DatasetError, "Method %s: the dataframe provided should have exactly 3 columns." %(CURR_FUNC_NAME)
    
    if 'userid' not in df.columns.values[0] or 'itemid' not in df.columns.values or 'rating' not in df.columns.values:
        raise DatasetError, "Method %s: the dataframe provided should have 3 columns named \"userid\", \"itemid\" and \"rating\"." %(CURR_FUNC_NAME)
    
    # Load the dataframe values in a hash. It will make life easier.
    
    ratingsHash = {}
    for _row_indx, (userid, itemid, rating) in df.iterrows():
        ratingsHash[(userid, itemid)] = rating
    
    # We now need to make our m x n sparse array.
    
    rowIndex = 'userid' if axis == 0 else 'itemid'
    columnIndex = 'itemid' if axis == 0 else 'userid'
    uniqueRows = df[rowIndex].unique()
    uniqueCols = df[columnIndex].unique()
    
    sparseArr = np.zeros((len(uniqueRows), len(uniqueCols)))    # zerofill initially
    for i in range(len(uniqueRows)):
        for j in range(len(uniqueCols)):
            if (uniqueRows[i], uniqueCols[j]) in ratingsHash:
                sparseArr[i][j] = ratingsHash[(uniqueRows[i], uniqueCols[j])]

    # Compute the covariance matrix
    print "sparseArr shape: " + str(sparseArr.shape)
    covMat = np.cov(sparseArr.T) 
    
    # A compressed representation is needed because we need to center sparse data.
     
    csr_rep = lil_matrix(sparseArr).tocsr() 
    for c in range(csr_rep.shape[axis]):
        sparseArr[c, :][sparseArr[c, :].nonzero()] -= np.mean(csr_rep.getcol(c).data)
    
    # Find eigenvalues, compute and return k-dimensional embedding
    print "covMat shape: " + str(covMat.shape)
    eigenVals, eigenVecs = eig(covMat)
    
    # Re-arrange eigenvectors so that you get the most significant components first.
    eigenVecs = eigenVecs[:, np.argsort(-eigenVals)][:, :k]
    return sparseArr.dot(eigenVecs) 

    
if __name__ == '__main__':
    
    os.chdir('../')
    ratings_train = pd.load('proc_data/ratings_train.pda')
    pca(ratings_train[['userid', 'itemid', 'rating']], 100, 0).real
    pca2(ratings_train[['userid', 'itemid', 'rating']], 100, 1).real