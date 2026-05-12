"""
Implementation based on: 
- https://medium.com/we-talk-data/how-can-i-use-knn-and-random-forest-models-in-pytorch-6083f5ef370a
- https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/random_forest.py
- Assistance from Generative AI
"""

import torch

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, num_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.tree = None

    def fit(self, X, y):
        if self.num_features is None:
            self.num_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(torch.unique(y))

        # stop conditions for recursion
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = self._most_common_label(y) # finds most frequent class
            return {"leaf": True, "value": leaf_value}

        # select random feature subset
        feat_idxs = torch.randperm(n_features)[:self.num_features] 

        # find the best feature and threshold to split on
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:
            return {"leaf": True, "value": self._most_common_label(y)}

        left_idxs = X[:, best_feature] <= best_thresh
        right_idxs = X[:, best_feature] > best_thresh

        if left_idxs.sum() == 0 or right_idxs.sum() == 0:
            return {"leaf": True, "value": self._most_common_label(y)}

        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        # return node
        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_thresh,
            "left": left,
            "right": right
        }

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        # loop thru every possible split
        for feature in feat_idxs:
            thresholds = torch.unique(X[:, feature])
            for thresh in thresholds:
                gain = self._information_gain(y, X[:, feature], thresh)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = thresh

        return split_idx, split_thresh

    # tests gini impurity improvement from a given split
    def _information_gain(self, y, feature_col, threshold):
        parent_loss = self._gini(y) # gini impurity before splitting (baseline)

        # boolean split masks: for each value in a feature column, check against threshold
        left_idxs = feature_col <= threshold
        right_idxs = feature_col > threshold

        if left_idxs.sum() == 0 or right_idxs.sum() == 0: # discard splits if all data goes to one side
            return 0

        n = len(y) # total samples
        n_l, n_r = left_idxs.sum(), right_idxs.sum() # number going left/right in split

        # calculate gini impurities for each split
        impurity_l = self._gini(y[left_idxs])
        impurity_r = self._gini(y[right_idxs])

        child_loss = (n_l / n) * impurity_l + (n_r / n) * impurity_r # average impurity after split
        return parent_loss - child_loss

    def _gini(self, y):
        classes = torch.unique(y)
        impurity = 1.0
        for c in classes:
            p = torch.sum(y == c).float() / len(y)
            impurity -= p ** 2
        return impurity

    def _most_common_label(self, y):
        values, counts = torch.unique(y, return_counts=True)
        return values[torch.argmax(counts)]

    def predict(self, X):
        preds = []
        for x in X:
            preds.append(self._traverse_tree(x, self.tree))
        return torch.tensor(preds)

    def _traverse_tree(self, x, node):
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]: # check split condition (value of feature used for split <= threshold)
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, min_samples_split=2, num_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.num_features = num_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                num_features=self.num_features
            )

            # create bootstrap samples
            idxs = torch.randint(0, X.shape[0], (X.shape[0],)) 
            X_sample = X[idxs]
            y_sample = y[idxs]

            tree.fit(X_sample, y_sample) # train decision tree on sampled data
            self.trees.append(tree)

    # combine predictions from all trees and return most common labels
    def predict(self, X):
        tree_preds_list = []
        for tree in self.trees:
            tree_preds_list.append(tree.predict(X))

        tree_preds = torch.stack(tree_preds_list)

        #return torch.mode(tree_preds, dim=0).values
        preds = []
        for i in range(tree_preds.shape[1]):
            vals, counts = torch.unique(tree_preds[:, i], return_counts=True)
            preds.append(vals[torch.argmax(counts)])

        return torch.tensor(preds)
