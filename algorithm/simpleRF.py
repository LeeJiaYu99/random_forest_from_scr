import random
import numpy as np

"""
Random Forests From Scratch: 
Learn how to build a random forest from scratch with python.

Reference & Code by:
tannercarbonati@gmail.com
https://carbonati.github.io/posts/random-forests-from-scratch/
"""

def entropy(p):
    """
    Measurement of impurity.
    """
    if p == 0:
        return 0
    elif p == 1:
        return 0
    else:
        return - (p * np.log2(p) + (1 - p) * np.log2(1-p))

def information_gain(left_child, right_child):
    """
    Comparison of entropy before split and after a split.
    """
    parent = left_child + right_child
    p_parent = parent.count(1) / len(parent) if len(parent) > 0 else 0
    p_left = left_child.count(1) / len(left_child) if len(left_child) > 0 else 0
    p_right = right_child.count(1) / len(right_child) if len(right_child) > 0 else 0
    IG_p = entropy(p_parent)
    IG_l = entropy(p_left)
    IG_r = entropy(p_right)
    return IG_p - len(left_child) / len(parent) * IG_l - len(right_child) / len(parent) * IG_r

def draw_bootstrap(X_train, y_train):
    """
    Bootstrapping (sampling with replacement) to inject randomness to each trees.
    Some of the samples may not be selected, these samples are called out-of-bag samples.
    """
    bootstrap_indices = list(np.random.choice(range(len(X_train)), len(X_train), replace = True))
    oob_indices = [i for i in range(len(X_train)) if i not in bootstrap_indices]
    X_bootstrap = X_train.iloc[bootstrap_indices].values
    y_bootstrap = y_train[bootstrap_indices]
    X_oob = X_train.iloc[oob_indices].values
    y_oob = y_train[oob_indices]
    return X_bootstrap, y_bootstrap, X_oob, y_oob

def oob_score(tree, X_test, y_test):
    """
    From bootstrapping process, OOB score for each tree is computed.
    These scores are then averaged to estimate how accurate the random forest performs. (leave-one-out cross validation).
    """
    mis_label = 0
    for i in range(len(X_test)):
        pred = predict_tree(tree, X_test[i])
        if pred != y_test[i]:
            mis_label += 1
    return mis_label / len(X_test)

def find_split_point(X_bootstrap, y_bootstrap, max_features):
    """
    The best split point is determined through computation of information gain of the possible split choices.
    1. select m features at random. 
    2. For each feature selected, iterate through each value in the bootstrapped dataset and compute the information gain.
    3. Return a dictionary of 'feature index', 'split point', 'left child node' and 'right child node' from the value that gives the highest information gain.
    """
    feature_ls = list()
    num_features = len(X_bootstrap[0])

    while len(feature_ls) <= max_features:
        feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in feature_ls:
            feature_ls.extend(feature_idx)

    best_info_gain = -999
    node = None
    for feature_idx in feature_ls:
        for split_point in X_bootstrap[:,feature_idx]:
            left_child = {'X_bootstrap': [], 'y_bootstrap': []}
            right_child = {'X_bootstrap': [], 'y_bootstrap': []}

            # split children for continuous variables
            if type(split_point) in [int, float]:
                for i, value in enumerate(X_bootstrap[:,feature_idx]):
                    if value <= split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])
            # split children for categoric variables
            else:
                for i, value in enumerate(X_bootstrap[:,feature_idx]):
                    if value == split_point:
                        left_child['X_bootstrap'].append(X_bootstrap[i])
                        left_child['y_bootstrap'].append(y_bootstrap[i])
                    else:
                        right_child['X_bootstrap'].append(X_bootstrap[i])
                        right_child['y_bootstrap'].append(y_bootstrap[i])

            split_info_gain = information_gain(left_child['y_bootstrap'], right_child['y_bootstrap'])
            if split_info_gain > best_info_gain:
                best_info_gain = split_info_gain
                left_child['X_bootstrap'] = np.array(left_child['X_bootstrap'])
                right_child['X_bootstrap'] = np.array(right_child['X_bootstrap'])
                node = {'information_gain': split_info_gain,
                        'left_child': left_child,
                        'right_child': right_child,
                        'split_point': split_point,
                        'feature_idx': feature_idx}

    return node

def terminal_node(node):
    """
    Stop splitting nodes in a tree and output a terminal node.
    """
    y_bootstrap = node['y_bootstrap']
    pred = max(y_bootstrap, key = y_bootstrap.count)
    return pred


def split_node(node, max_features, min_samples_split, max_depth, depth):
    """
    On a single tree,
    1. Given a node, store the left and right children as left_child & right_child and remove them from the original dictionary.
    2. Check number of samples in each children. If empty, that means the best split for that node was unable to differentiate the 2 classes, call terminal node.
    3. Check if current depth has reached the max_depth, call terminal node if reached.
    For both left and right child node:
    4. Check if samples in the children at the current node is less than min_sample_split, call terminal node if less than.
    5. If samples are more than mn_sample_split, feed the node into find_split_point again and repeat the steps until each branch has a terminal node.
    """
    left_child = node['left_child']
    right_child = node['right_child']

    del(node['left_child'])
    del(node['right_child'])

    if len(left_child['y_bootstrap']) == 0 or len(right_child['y_bootstrap']) == 0:
        empty_child = {'y_bootstrap': left_child['y_bootstrap'] + right_child['y_bootstrap']}
        node['left_split'] = terminal_node(empty_child)
        node['right_split'] = terminal_node(empty_child)
        return

    if depth >= max_depth:
        node['left_split'] = terminal_node(left_child)
        node['right_split'] = terminal_node(right_child)
        return node

    if len(left_child['X_bootstrap']) <= min_samples_split:
        node['left_split'] = node['right_split'] = terminal_node(left_child)
    else:
        node['left_split'] = find_split_point(left_child['X_bootstrap'], left_child['y_bootstrap'], max_features)
        split_node(node['left_split'], max_depth, min_samples_split, max_depth, depth + 1)
    if len(right_child['X_bootstrap']) <= min_samples_split:
        node['right_split'] = node['left_split'] = terminal_node(right_child)
    else:
        node['right_split'] = find_split_point(right_child['X_bootstrap'], right_child['y_bootstrap'], max_features)
        split_node(node['right_split'], max_features, min_samples_split, max_depth, depth + 1)

def build_tree(X_bootstrap, y_bootstrap, max_depth, min_samples_split, max_features):
    """
    Constuct the trees based on the paramters:
    - n_estimators: (int) The number of trees in the forest.
    - max_features: (int) The number of features to consider when looking for the best split.
    - max_depth: (int) The maximum depth of the tree.
    - min_samples_split: (int) The minimum number of samples required to split an internal node.
    """
    root_node = find_split_point(X_bootstrap, y_bootstrap, max_features)
    split_node(root_node, max_features, min_samples_split, max_depth, 1)
    return root_node

def random_forest(X_train, y_train, n_estimators, max_features, max_depth, min_samples_split):
    """
    To build a tree, first, a bootstrap sample of size n is drawn from the data.
    Then build the tree from the bootstrapped sample by repeatedly:
    1. sample m features for bagging.
    2. compute the information gain for each possible value among the bootstrapped data and m features.
    3. split the node into 2 children nodes,
    until each node consists of 1 class only or minimum node size specified is reached.
    """
    tree_ls = list()
    oob_ls = list()
    for i in range(n_estimators):
        X_bootstrap, y_bootstrap, X_oob, y_oob = draw_bootstrap(X_train, y_train)
        tree = build_tree(X_bootstrap, y_bootstrap, max_features, max_depth, min_samples_split)
        tree_ls.append(tree)
        oob_error = oob_score(tree, X_oob, y_oob)
        oob_ls.append(oob_error)
    print("OOB estimate: {:.2f}".format(np.mean(oob_ls)))
    return tree_ls

def predict_tree(tree, X_test):
    """
    For inferencing, use split points of trained random forest (in a nested dictionary) to perform splitting and constructing of treeson top of test data.
    """
    feature_idx = tree['feature_idx']

    if X_test[feature_idx] <= tree['split_point']:
        if type(tree['left_split']) == dict:
            return predict_tree(tree['left_split'], X_test)
        else:
            value = tree['left_split']
            return value
    else:
        if type(tree['right_split']) == dict:
            return predict_tree(tree['right_split'], X_test)
        else:
            return tree['right_split']
        
def predict_rf(tree_ls, X_test):
    """
    Start inferencing process and output final prediction.
    """
    pred_ls = list()
    for i in range(len(X_test)):
        ensemble_preds = [predict_tree(tree, X_test.values[i]) for tree in tree_ls]
        final_pred = max(ensemble_preds, key = ensemble_preds.count)
        pred_ls.append(final_pred)
    return np.array(pred_ls)