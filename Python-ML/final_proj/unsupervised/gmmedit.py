'''
Created on Dec 16, 2012

@author: sar
'''
import numpy as np
#from sklearn import mixture
import gmm_diag2_forJason as proj


if __name__ == '__main__':
    np.random.seed(1)
    g = proj.GMM(n_components=2, covariance_type='full')
    # Generate random observations with two modes centered on 0
    # and 10 to use for training.
    obs = np.concatenate((np.random.randn(100, 5), 10 + np.random.randn(300, 5)))
    print obs.shape
    g.fit(obs)
    print "weights = ", np.round(g.weights_, 2)
    print "means = ", np.round(g.means_, 2)
    print "covars = ", np.round(g.covars_, 2) #doctest: +SKIP
    #g.predict([[0], [2], [9], [10]])
    #print np.round(g.score([[0], [2], [9], [10]]), 2)
    # Refit the model on new data (initial parameters remain the
    # same), this time with an even split between the two modes.
    #g.fit(20 * [[0]] +  20 * [[10]])
    #print np.round(g.weights, 2)
    
    pass