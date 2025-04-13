# Random Forest from Scratch
## 🌲 Background
This repository contains a simple implementation of random forest algorithm inspired by [this blog post](https://carbonati.github.io/posts/random-forests-from-scratch/). While there is existing libraries, such as [RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) in sklearn framework that is ready to use by direct integration into Python projects, I decided to learn the algorithm from the ground up to understand the inner workings of the model.

Note that the core implementation is in <mark>algorithm</mark> folder. Special credit to [carbonati](https://github.com/carbonati/machine-learning) for the amazing source code. The [titanic dataset](https://www.kaggle.com/competitions/titanic/data) from Kaggle is used to demonstrate the functionality of the custom random forest model.

## 🌲 Introduction to Random Forest
Random forest is an ensemble learning algorithm based on decision tree.  It evolves from the basic decision tree model by building multiple trees and aggregating their predictions. The bagging and bootstrapping technique used in random forest result in more robust model that overfitting is prevented and variance is reduced. 

## 🌲 Basic Form of Random Forest
Tree based-algorithms construct trees using nodes (leaves) and edges (branches). 
- The root node is the first node at the top of the tress, where the tree building process begins.

- Terminal nodes (leaf nodes) are the final nodes without further child nodes. 

- The rest of the nodes which are neither root nor terminal nodes are internal nodes as a result of splitting process. 

Each split creates two child nodes, namely left and right node. The nodes are connected by edges.

## 🌲 Random Forest Algorithm
### Bootstrapping
In each node, bootstrapping is performed to randomly select samples with replacement for training the tree. This process left out some samples and they are referred to as out-of-bag samples that could be used for validation. The out-of-bag error obtained from validation provides an additional metric to access the performance of the model. 

### Aggregation bootstrapping (Bagging)
The bootstrapping process is repeated accross different trees to achieve aggregation bootstrapping (also known as bagging). Bagging employs a 'voting' system (ensemble method) to combine preditions of multiple trees. This helps to reduce variance, remove noise and prevent overfitting which is an issue in individual decision tree. As bagging would look at multiple uncorrelated trees with varying bootstrap samples and features, random forest is able to capture the variability as well as interactions between variables for more effective generization.

### Splitting criteria
During the splitting process, only one node (either left or right node) is chosen for further splitting. How does the tree decide which node to split? 

It is based on which split provides the higheset information gain, a measure of how much a split improves the purity of the node. There are two commonly used criteria for calculating information gain:

1.  Gini index = $$1 - \Sigma (p_i)^2 $$ 
2. entropy = $$- \Sigma p_i log_2(p_i) $$

Where $p_i$ is the probability of each class. In this source code, we specify the model to use entropy.

 They are both measures of impurity or uncertainty in a node. The goal is to minimize the entropy (or Gini index) after each split, which results in the most informative splits.

## 🌲 Parameters for Random Forest
If a node does not achieve sufficient purity, the tree continues to split. This would lead to excessively deep trees that are computationally expensive and prone to overfitting. To address this, we would need these parameters to control the splits:
-  <mark>min_sample_split</mark>: The minimum number of samples in the node before the node can be split, else the node should stop splitting.  

- <mark>max_depth</mark>: The maximum depth to which the tree can grow.

- <mark>max_features</mark>: The maximum number of features that can be considered for splitting at each node during bootstrapping. 

- <mark>n_estimates</mark>: The number of trees in the forest. 

Choosing the optimal values for these parameters depends on the characteristics of the dataset and the trade-off between performance and computational efficiency.

## 🌲 Application
Random forest is effective with high-dimensional and non-linear datasets. It can be applied in classification and regression problem:

- In classification, the Random Forest algorithm outputs the most frequent class (the mode) predicted by the individual trees.

- In regression, the algorithm outputs the average of the predictions from the individual trees.
