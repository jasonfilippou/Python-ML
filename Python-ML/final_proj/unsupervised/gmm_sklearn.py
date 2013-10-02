from sklearn import mixture
import os
import numpy as np
import pickle as pkl

def loadData(datadirectory= '/media/Common/Eclipse/GMM/caltech101/training/'):
    DIR_categories=os.listdir(datadirectory);
    imFeatures=[]
    imLabels=[]
    labelNames = []
    labelCounts = []
    i=0;
    for cat in DIR_categories:
        if os.path.isdir(datadirectory+ cat):
            i=i+1;
            DIR_image=os.listdir(datadirectory+ cat +'/');
            count = 0;
            for im in DIR_image:
                if (not '._image_' in im):
                    F = np.genfromtxt(datadirectory+cat+'/'+im, delimiter=' ');
                    F = np.reshape(F,21*28);
                    #F = np.mat(F);
                    F = F.tolist();
                    imFeatures.append(F);
                    imLabels.append(i);
                    count = count + 1
            labelNames.append(cat);
            labelCounts.append(count);
    return np.array(imFeatures), np.array(imLabels), labelNames, labelCounts

def loadTestData(datadirectory = '/media/Common/Eclipse/GMM/caltech101/training/'):
    #DIR_categories=os.listdir(datadirectory);
    imFeatures=[]
    imLabels=[]
    labelNames = []
    labelCounts = []
    i=0;
    for cat in ['Motorbikes','Faces_easy']:
        if os.path.isdir(datadirectory+ cat):
            i=i+1;
            DIR_image=os.listdir(datadirectory+ cat +'/');
            count = 0;
            for im in DIR_image:
                if (not '._image_' in im):
                    F = np.genfromtxt(datadirectory+cat+'/'+im, delimiter=' ');
                    F = np.reshape(F,21*28);
                    #F = np.mat(F);
                    F = F.tolist();
                    imFeatures.append(F);
                    imLabels.append(i);
                    count = count + 1
            labelNames.append(cat);
            labelCounts.append(count);
    return np.array(imFeatures), np.array(imLabels), labelNames, labelCounts

if __name__ == '__main__':
    nclusters = 101
    trainX,trainY,trainNames,trainCounts = loadData()
    val_features, truelabels, names_, counts_ = loadData('/media/Common/Eclipse/GMM/caltech101/validation/')
    pkl.dump(truelabels,open('results/truelabels.out','wb'))
    g1 = mixture.GMM(n_components=nclusters)  #diagonal
    g1.fit(trainX)
    predlabels1 = g1.predict(val_features)
    predproba1 = g1.predict_proba(val_features)
    #print predlabels1
    pkl.dump(predlabels1,open('results/predlabels_diag.out','wb'))
    pkl.dump(predproba1,open('results/predproba_diag.out','wb'))
    
    g2 = mixture.GMM(n_components=nclusters,  cvtype='full') #full
    g2.fit(trainX)
    predlabels2 = g2.predict(val_features)
    predproba2 = g2.predict_proba(val_features)
    #print predlabels2
    pkl.dump(predlabels2,open('results/predlabels_full.out','wb'))
    pkl.dump(predproba2,open('results/predproba_full.out','wb'))
    
    #try with OVA
    
    