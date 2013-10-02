"""
Utilities for scoring splits in a decision tree
"""
import numpy as np
from Node import Split

def gini(p):
    """ Compute gini index for vector p"""
    return np.sum(p*(1-p))

def scoreit(left_p, right_p, nleft, nright, fn=gini):
    """
    Compute score for a split from computed proportions and number of examples.

    Arguments:
    left_p -- class proportions for left child
    right_p -- class proportions for right child
    nleft -- number of examples in left child
    nright -- number of examples in right child
    fn -- scoring function (default: gini index)

    Returns:
    score for split
    """
    n = float(nleft + nright)

    # compute gini index for left child
    fleft = fn(left_p)

    # compute gini index for right child
    fright = fn(right_p)

    # reutrn weighted average
    return nleft/n * fleft + nright/n * fright

def score_feature_value(df, feature_name, feature_value, label_name):
    """
    Score a split condition (feature_name==feature_value)

    Arguments:
    df -- pandas.DataFrame to use for scoring
    feature_name -- column in df to use for splitting
    feature_value -- the feature value to split on
    label_name -- the column in df containing class labels

    Returns:
    Score for split
    """

    # compute number of examples for each feature value
    cnts = df.groupby([feature_name, label_name]).size()

    # get lists of feature values and class labels
    feature_values, class_labels = cnts.index.levels

    # only one value here so can't split (return inf)
    if (len(feature_values)==1):
        return np.inf

    # get indices into feature_values and class_labels of each entry in cnts
    # read pandas documentation of hierarchical indexing
    feature_index, label_index = cnts.index.labels

    # a function to compute class proportions over subsets of cnts vector
    get_props = lambda indx: np.array([sum(cnts[indx][label_index[indx]==j]) for j in range(len(class_labels))])/float(sum(cnts[indx]))

    # index into feature_values where value to score is located
    i = np.where(feature_values == feature_value)[0]

    # index into cnts where the counts for value to score are located
    indx = np.where(feature_index == i)[0]

    # compute label proportions for examples that satisfy the split condition
    left_props = get_props(indx)

    # compute the number of examples that satisfy the split condition
    nleft = sum(cnts[indx])

    # index into cnts where the counts for other feature values are located
    indx = feature_index != i

    # compute the label proportions for examples that don't satisfy the split condition
    right_props = get_props(indx)

    # compute the number of examples that don't satisfy the split condition
    nright = sum(cnts)-nleft

    # score the split
    res = scoreit(left_props, right_props, nleft, nright)
    return res

def get_candidate_values(df, feature_name):
    """
    Find candidate splits for given feature.

    Arguments:
    df -- pandas.DataFrame to use to find candidate values for splitting
    feature_name -- column name to find cnadidate values for

    Returns:
    Tuples (feature_name, feature_value) for candidate splitting values
    """

    # use groupby to find the values for this feature
    cnts = df.groupby([feature_name]).size()

    # only one value, so can't split on this
    if (len(cnts)==1):
        return []

    # get the feature values
    feature_values = cnts.index.values

    # return tuples
    return [(feature_name, feature_value) for feature_value in feature_values]

def best_split(df, features_to_use, label_name):
    """
    Find the best split for examples

    Arguments:
    df -- pandas.DataFrame to use to find best split
    features_to_use -- features to consider when splitting
    label_name -- column in df containing class labels

    Returns:
    node -- an object of class Split describing split condition (None if no split possible)
    features_left -- features to check for splitting in left child (None if no split possible)
    score -- the score of best split (None if no split possible)
    """

    # get the names of the features to use
    feature_names = df.columns.values[features_to_use]

    # list of candidate splits
    candidate_values = []

    # add candidate values for each feature
    for feature_name in feature_names:
        candidate_values += get_candidate_values(df, feature_name)

    # no candidate values, can't split here
    if (len(candidate_values)==0):
        return None, None, None


    scores = np.inf * np.ones(len(candidate_values))
    for j in range(len(candidate_values)):
        feature_name = candidate_values[j][0]
        feature_value = candidate_values[j][1]
        res = score_feature_value(df, feature_name, feature_value, label_name)
        scores[j] = res

    # find the index of best scoring split
    minIndx = np.argmin(scores)

    # remove this feature from the set of features to use
    # the left child doesn't need to look at this feature again
    feature_index = np.where(feature_names==candidate_values[minIndx][0])[0]
    features_left = np.delete(features_to_use, feature_index)
    return Split(candidate_values[minIndx][0], candidate_values[minIndx][1]), features_left, scores[minIndx]
