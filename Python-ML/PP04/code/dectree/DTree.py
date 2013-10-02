"""
An implementation of Decision Tree classification algorithm. Produces binary trees,
for categorical features. Handles multiple classes (not just binary).
By default, uses the Gini index to select node splits.
"""
import numpy as np
import inspect
from util.mlExceptions import DatasetError, LogicalError
from scoring import best_split, weightedGroupBy
from Node import Inner, Leaf

class DTree(object):
    """
    A binary decision tree classifier. Handles only categorical features, but can handle multiple classes.
    Uses the Gini index to select node splits.

    Attributes:
    traindat -- a pandas.DataFrame used for training
    label_name -- the name of column in traindat that contains labels
    depth -- the depth of the tree
    optimal_depth -- the optimal depth to use in prediction (set by tune method)
    root -- reference to Node object at tree root
    features_to_use -- the index of columns in traindat usable as features
    weights --- a weight vector used by the AdaBoost algorithm
    """
    def __init__(self, traindat, label_name, weights):
        """
        Arguments:
        @param traindat -- a pandas.DataFrame to use for training
        @param label_name -- the name of the column in traindat where label is stored
        @param weights --- a weight vector representing the weights of the examples.
        """
        self.traindat = traindat
        self.label_name = label_name
        self.depth = None
        self.optimal_depth = None
        self.root = None
        self.weights = weights

        feature_names = self.traindat.columns.values
        features_to_use = np.arange(self.traindat.shape[1])[feature_names != self.label_name]
        feature_names = feature_names[features_to_use]

        # I believe that, in the following lines of code,
        # the tree designer is discarding irrelevant features.
        
        nvals = [len(traindat.groupby(feature_name).size().index) for feature_name in feature_names]
        keep_feature = np.array(nvals)>1
        self.features_to_use = features_to_use[keep_feature]        

    def __repr__(self):
        return self.root.__repr__()

    def train(self, maxdepth, verbose=0):
        """
        Train the decision tree. Sets the root attribute as a side effect

        Arguments:
        maxdepth -- the maximum depth to use in training
        verbose -- 0 produces no output

        """
        indexes = np.arange(self.traindat.shape[0])


        self.root, self.depth = self.train_helper(self.features_to_use, indexes, 0, maxdepth,verbose=verbose)
        self.optimal_depth = self.depth

    def split_func(self, indxs, features_to_use):
        """
        The function used to select the best splitting feature.

        Arguments:
        indxs -- rows in traindat to use for scoring
        features_to_use -- indices of columns in traindat to score

        Returns:
        split -- an object of class Split
        features_left -- the set of features to consider on the left child of tree
        score -- the score of the best split
        """
        return best_split(self.traindat.ix[indxs,:], features_to_use, self.label_name, self.weights)

    def get_class_props(self, indxs):
        """
        compute class proportions

        Arguments:
        indxs -- rows in traindat used to calculate proportions

        Returns:
        A hash table indexed by class labels containing class proportions
        """
        
        # TODO: Definitely need to define cnts and props in terms of weights.
        # Those weights you will need to pass as an argument.
        # See photograph.
        #cnts = self.traindat.ix[indxs,:].groupby(self.label_name).size()
        cnts = weightedGroupBy(self.traindat.ix[indxs, :], [self.label_name], self.weights)
        props = cnts/float(np.sum(cnts))
        class_labels = cnts.index
        return (dict(zip(class_labels, props)))

    def train_helper(self, features_to_use, current_indexes, depth, maxdepth, verbose=0):
        """
         Helper function for training

         Arguments:
         features_to_use -- indices into traindat to consider for splitting
         current_indexes -- rows of traindat in the current subtree
         depth -- current depth
         maxdepth -- maximum allowable depth in training
         verbose -- 0 produces no output

         Returns:
         node -- object of class Node
         depth -- the depth of the learned tree
        """
        if verbose>0:
            print 'Building tree at depth %d with %d examples and %d features' % (depth, len(current_indexes), len(features_to_use))
        nex_node = len(current_indexes)
        nex_total = self.traindat.shape[0]
        class_props = self.get_class_props(current_indexes)
        if verbose>0:
            print class_props
        #tmp=self.traindat.ix[current_indexes, features_to_use]
        #tmp[self.label_name]=self.traindat[self.label_name][current_indexes]
        #print tmp

        if len(features_to_use)==0 or depth==maxdepth or len(class_props)==1:
            out = Leaf(class_props, nex_node, nex_total, depth)
            if verbose>0:
                print 'returning:', out
            return out, depth
        split, features_left, best_score = self.split_func(current_indexes, features_to_use)
        if split is None or not np.isfinite(best_score):
            out = Leaf(class_props, nex_node, nex_total, depth)
            if verbose>0:
                print 'returning:', out
            return out, depth

        split_indxs = split.splitit(self.traindat, current_indexes)
        if verbose>0:
            print len(current_indexes), len(split_indxs)


        left,left_depth = self.train_helper(features_left, current_indexes[split_indxs], depth+1, maxdepth,verbose=verbose)
        right,right_depth = self.train_helper(features_to_use, current_indexes[split_indxs != True], depth+1, maxdepth,verbose=verbose)
        out_depth = left_depth if left_depth > right_depth else right_depth
        out = Inner(class_props, nex_node, nex_total, split, left, right, depth)
        if verbose>0:
            print 'returning:', out
        return out, out_depth

    def predict(self, dat, depth=None):
        """
        Predict labels

        Arguments:
        dat -- pandas.DataFrame to predict labels for
        depth -- maximum depth used in prediction (useful when tuning the tree), if None, uses optimal_depth property

        Returns:
        A numpy array with class labels
        """
        if depth is None:
            depth = self.optimal_depth
        nexamples = dat.shape[0]
        preds = np.empty(nexamples)
        indxs = np.arange(nexamples)
        return self.root.predict(depth, dat, indxs, preds)

    def tune(self, dat):
        """
        Tune the decision tree. Checks the depth that minimizes error rate on a tuning set.
        Sets the optimal_depth property as a side effect

        Arguments:
        dat -- pandas.DataFrame to use as a tuning set
        """
        depths = range(1,self.depth+1)
        get_rate = lambda depth: np.mean(dat[self.label_name] != self.predict(dat, depth))
        errors = np.array(map(get_rate, depths))
        indx = np.argmin(errors)
        print zip(depths,errors)
        self.optimal_depth = depths[indx]

    def generate_data(self, nexamples):
        """
        Generate data from the tree. Use with caution

        Arguments:
        nexamples -- number of examples to produce per leaf node

        Returns:
        A pandas.DataFrame with examples
        """
        dat = self.root.generate_data(nexamples)
        dat.index = np.arange(dat.shape[0])
        return dat

def get_tree(df, label_name, weights, maxdepth=10, verbose=0):
    """
    Get a decision tuned decision tree. Tuning is done on a held-aside tuning set with 20% of the data.

    Arguments:
    @param df -- pandas.DataFrame to use for training
    @param label_name -- column in df that contains labels
    @param maxdepth -- maximum depth to use when first constructing the tree
    @param verbose -- 0 produces no output
    @param weights: A weight vector, containing example weights. Used by AdaBoost.
    @return A decision tree
    @raise LogicalError if the weight vector size doesn't match the dataframe's row count.
    """
    
    # Sanity check arguments
    
    try:
        len(weights)
    except TypeError:
        raise LogicalError, "Method %s: the \"weights\" is not an iterable." %(inspect.stack()[0][3])
    if len(weights) != df.shape[0]:
        raise LogicalError, "Method %s: Length mismatch between weight vector and dataframe row count." %(inspect.stack()[0][3])
    
    nexamples = df.shape[0]
    indxs=np.random.permutation(np.arange(nexamples))
    
    train_df = df
    #ntrain = np.floor(nexamples * .8)
    #train_indxs = indxs[:ntrain]
    #tune_indxs = indxs[ntrain:]

    #train_df=df.ix[train_indxs,:]
    #tune_df=df.ix[tune_indxs,:]
    #train_df.index=np.arange(ntrain)
    #tune_df.index=np.arange(nexamples-ntrain)

    dt=DTree(train_df,label_name, weights)
    dt.train(maxdepth, verbose=verbose)
    #dt.tune(tune_df)
    return dt

