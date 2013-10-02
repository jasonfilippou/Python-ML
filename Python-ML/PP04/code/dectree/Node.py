"""
Classes implementing decision tree structure
"""
import numpy as np
import pandas as pd

class Split:
    """
    A split condition descriptor. Tests are of the form (feature_name == feature_value) producing binary trees.

    Attributes:
    feature_name -- the name of the feature used to split
    feature_value -- the value of the feature used to split
    """
    def __init__(self, feature_name, feature_value):
        """
        Arguments:
        feature_name -- name of the feature to split
        feature_value -- feature value to test
        """
        self.feature_name = feature_name
        self.feature_value = feature_value

    def __repr__(self):
        return '(%s=%s)' % (self.feature_name, str(self.feature_value))

    def get_value(self, matching=True):
        """
        Get a value from this split (used in generate_data)

        Arguments:
        matching -- boolean indicating if matching value is produced (if false a fake value is produced)

        Returns:
        A value
        """
        if matching:
            return self.feature_value

        offset = 10
        if type(self.feature_value) == type(''):
            offset = str(offset)

        return self.feature_value + offset

    def splitit(self, df, indxs):
        """
        Split a set of examples.

        Arguments:
        df -- a pandas.DataFrame to split
        indxs -- rows in df to split

        Returns:
        A boolean index indicating which rows match the split condition
        """
        return df.ix[indxs,self.feature_name] == self.feature_value

class Node(object):
    """
    Generic node class

    Attributes:
    class_props -- proportion of class labels corresponding to this node
    depth -- the node's depth in tree
    nex_node -- number of examples in training set that get to this node
    nex_total -- total number of examples in training set
    max_label -- the label of the majority of examples that landed in this node
    """
    def __init__(self, class_props, nex_node, nex_total, depth):
        """
        Arguments:
        class_props -- proportion of class labels corresponding to this node
        nex_node -- number of examples in training set that get to this node
        nex_total -- total number of examples in training set
        depth -- the node's depth in tree

        """
        self.class_props = class_props
        self.depth = depth
        self.nex_node = nex_node
        self.nex_total = nex_total

        maxIndx=np.argmax(np.array(class_props.values()))
        self.max_label = class_props.keys()[maxIndx]

    def __repr__(self):
        out = 'depth: %d, ' % self.depth
        out += ' '.join(['%s: %.2f' % item for item in self.class_props.iteritems()])
        out += ' (%d/%d) examples' % (self.nex_node, self.nex_total)
        return out

    def predict(self, depth, df, indxs, preds):
        """
         Predict labels using the class proportions in this node (max_label)
        """
        #print 'Done!'
        preds[indxs] = self.max_label
        return preds

class Inner(Node):
    """
    An inner node in the decision tree

    Attributes:
    split -- an object of class Split
    left -- the left child (examples that satisfy the split condition)
    right -- the right child (examples that do not satisfy the split condition)

    """
    def __init__(self, class_props, nex_node, nex_total, split, left, right, depth):
        """
        Arguments:
        class_props -- proportion of class labels for examples that landed in this node
        nex_node -- number of examples that landed in this node
        nex_total -- total number of examples in training set
        split -- object of class Split
        left -- the left child (examples that satisfy split condition)
        right -- the right child (examples that don't satisfy split condition)
        depth -- the depth of this node in the tree
        """
        super(Inner, self).__init__(class_props, nex_node, nex_total, depth)
        self.split = split
        self.left = left
        self.right = right

    def __repr__(self):
        out = ('  ' * self.depth) + 'InnerNode %s: ' % (self.split.__repr__())
        out += super(Inner, self).__repr__() + '\n'
        out += 'Y: ' + self.left.__repr__()
        out += 'N: ' + self.right.__repr__()

        return out

    def generate_data(self, nexamples=10, path=[]):
        """
        Generate data from this node

        Arguments:
        nexamples -- number of examples to generate
        path -- the path through tree to get to this node (tuples (['y'|'n'], Split object))

        Returns
        pandas.DataFrame of generated examples
        """

        ## get data from left child
        df = self.left.generate_data(nexamples, [('y', self.split)]+path)

        ## append rows from right child
        df = df.append(self.right.generate_data(nexamples, [('n',self.split)]+path))

        ## fill in missing data
        return df.fillna(10)


    def get_feature_names(self):
        """
        Get feature names for subtree rooted at this node

        Returns
        Set of feature names
        """
        out = set([self.feature_name])
        for child in self.children.itervalues():
            out = out | child.get_feature_names()
        return out

    def predict(self, depth, df, indxs, preds):
        """
        Predict labels

        Arguments:
        depth -- depht used for prediction
        df -- pandas.DataFrame of examples
        indxs -- which rows of df to predict
        preds -- running vector of predictions

        Returns:
        updated vector of predictions
        """
        #print len(indxs), self.depth, depth
        #print self

        # no examples left to predict
        if (len(indxs)==0):
            return preds

        # we are at prediction depth, predict according to this node
        if (depth==self.depth):
            return super(Inner, self).predict(depth, df, indxs, preds)

        # split the DataFrame according the this node's split condition
        split_indxs = self.split.splitit(df, indxs)

        # get predictions from left child
        preds = self.left.predict(depth if depth is not None else None, df, indxs[split_indxs], preds)

        # return with updated predictions from right child
        return self.right.predict(depth if depth is not None else None, df, indxs[split_indxs != True], preds)


class Leaf(Node):
    """
    A leaf node in decision tree
    """
    def __repr__(self):
        out = ('  ' * self.depth) + 'LeafNode '
        out += super(Leaf, self).__repr__() + '\n'
        return out

    def get_feature_names(self):
        """
        Return a set of feature names for subtree rooted at this node

        Returns:
        empty set
        """
        return set([])

    def generate_data(self, nexamples=10, path=[]):
        """
        Generate data according to label proportions of this leaf node and fill in features from path

        Arguments:
        nexamples -- number of examples to generate
        path -- the path through tree to get to this node (tuples (['y'|'n'], Split object))

        Returns:
        pandas.DataFrame of data
        """
        breaks=[0] + list(np.cumsum(self.class_props.values()))
        flips = np.random.random_sample(nexamples)
        idx = np.array(pd.cut(flips, breaks).labels)
        out = pd.DataFrame({'label' : pd.Series(np.array(self.class_props.keys())[idx])})
        if len(path) > 0:
            for dir,split in path:
                if split.feature_name not in out:
                    out[split.feature_name]=split.get_value(dir=='y')
        return out


