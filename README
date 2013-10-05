Python-ML
=========

- Overview: 

Python-ML is a set of programming assignments which were requirements for the graduate-level Machine Learning
course (CMSC726) offered at the University of Maryland during the Fall 2012 semester. All assignments were 
implemented in Python and include documentation in the form of iPython notebooks. 

- Specifics:

PP01 is an implementation of a decision tree classifier as well as various nearest neighbor classifiers. Those classifiers 
are then applied to the common ML task of spam e-mail classification.

PP02 concerns multi-class classification using linear classifiers. The One-Versus-All and All-Versus-All
classification schemes are considered, whereas on the classifier side the Averaged Perceptron and Support Vector
Machine are implemented.

PP03's theme is collaborative filtering and ensemble methods (Bagging, Random Forests, Boosting). 

PP04 concerns Dimensionality Reduction, in particular an implementation of Principal Component Analysis (PCA)
and its applications on classification.

Finally, final_proj (the course's final project) was a team effort accomplished with the help of Sarthak Grover
(sgrover [AT] gatech [DOT] edu). Our aim with this project was to understand how "accurate" different clustering
algorithms are, i.e whether the clusters produced have some sort of intuitive meaning / interpretation. To accomplish this,
we used the well-known Computer Vision dataset Caltech 101. We extracted three different kinds of features (intensities
and SIFT with codebook sizes of 700 and 1000). We subsequently stripped the labels from the examples. 
Finally, we ran K-means and the Gaussian Mixture Model on all different features to produce clusters. Our evaluation
methodology consisted of examining the "purity" of each cluster with respect to the ground truth labels of the examples
it was assigned: if the in-cluster distribution of labels was skewed (i.e the cluster had low entropy when considering
the labels of the examples assigned to it) then we concluded that it was a "pure" cluster. Intuitively, we wanted to
understand what kind of images the clusters grouped together: did the features "push" the algorithms towards clusters
that represented the actual objects, or was it perhaps the case that due to image characteristics (sharp edges,
confusing backgrounds, etc) the clusters were not representative of single object classes? Perhaps unsurprisingly,
we found the combination of the GMM and SIFT features with k=1000 to be the most effective under those general
evaluation criteria. As a side product of the work, we experimented with various values for the expected cluster 
number on Caltech 101 (C = 2, 3, 5 and 10) to try to understand how the different labels are distributed across clusters.
Our results are plotted in the included documentation.

- Documentation:

To run the project-specific documentations, you need to have iPython (http://ipython.org/) installed. Once iPython
is installed, navigate to any sub-directory of the repo through the shell of your choice and type:

ipython notebook --pylab inline

This should work for both Windows and Unix-based systems provided that the ipython executable is visible.

- Licensing:

This project is licensed under the "Academic Free License" (AFL 3.0). Refer to the file LICENSE for more details.
