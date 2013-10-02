'''
Created on Dec 17, 2012

@author: jason
'''

import os
import pickle as pkl
from scipy.io import loadmat

def storeGradientHash():
        
    DIR_categories=os.listdir('input_data/caltech_training_gradients/');       # list and store all categories (classes) 
    i=0;  
    trueLabelHash = dict()                                                              # this integer will effectively be the class label in our code (i = 1 to 101)                                           
    for cat in DIR_categories:                                          # loop through all categories
        if os.path.isdir('input_data/caltech_training_gradients/'+ cat):   
            i=i+1;                                             # i = current class label
            trueLabelHash[i] = cat
                                
    print trueLabelHash
    pkl.dump(trueLabelHash, open('proc_data/gradientLabelAssociations.pkl', 'wb'))

def storeSIFTHash():
    
    DIR_categories=os.listdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000');         
    trueLabelHash = dict()
    i=0       
    for cat in DIR_categories:   
        if os.path.isdir('input_data/caltech101_SIFT/dense_bow/oneForAll_nr1_K1000/' + cat):
            i = i + 1
            trueLabelHash[i] = cat
            
    pkl.dump(trueLabelHash, open('proc_data/SIFTLabelAssociations.pkl', 'wb'))
    
if __name__ == "__main__":
    
    os.chdir("../")
    storeGradientHash()
    storeSIFTHash()