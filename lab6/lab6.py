import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------- A1: Entropy ----------
def entropy(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

# ---------- Equal Width Binning ----------
def equal_width_binning(X, bins=4):
    X_binned = X.copy()
    for col in range(X.shape[1]):
        col_min = X[:, col].min()
        col_max = X[:, col].max()
        width = (col_max - col_min) / bins
        X_binned[:, col] = np.floor((X[:, col] - col_min) / width)
    return X_binned.astype(int)

# ---------- A2: Gini ----------
def gini(y):
    counts = np.bincount(y)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

# ---------- A3: Information Gain ----------
def information_gain(X_col, y):
    total_entropy = entropy(y)
    values = np.unique(X_col)

    weighted_entropy = 0
    for v in values:
        y_subset = y[X_col == v]
        weighted_entropy += (len(y_subset) / len(y)) * entropy(y_subset)

    return total_entropy - weighted_entropy

def best_feature(X, y):
    gains = [information_gain(X[:, i], y) for i in range(X.shape[1])]
    return np.argmax(gains)

# ---------- A5: Simple Decision Tree ----------
class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.label = label

def build_tree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1 or depth == max_depth:
        return Node(label=Counter(y).most_common(1)[0][0])

    feat = best_feature(X, y)
    values = np.unique(X[:, feat])

    if len(values) == 1:
        return Node(label=Counter(y).most_common(1)[0][0])

    split_val = values[0]

    left_idx = X[:, feat] == split_val
    right_idx = X[:, feat] != split_val

    left = build_tree(X[left_idx], y[left_idx], depth+1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth+1, max_depth)

    return Node(feature=feat, value=split_val, left=left, right=right)

def predict_tree(node, x):
    if node.label is not None:
        return node.label
    if x[node.feature] == node.value:
        return predict_tree(node.left, x)
    else:
        return predict_tree(node.right, x)

# ---------- Load Dataset ----------
data = pd.read_csv("cmu_mosi_numeric.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# ---------- Binning ----------
X_binned = equal_width_binning(X, bins=4)

# ---------- A1 & A2 ----------
print("Entropy:", entropy(y))
print("Gini:", gini(y))

# ---------- A3 ----------
root_feature = best_feature(X_binned, y)
print("Best Feature (Root):", root_feature)

# ---------- A5 ----------
tree = build_tree(X_binned, y)

# ---------- A6 (Sklearn visualization for clarity) ----------
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_binned, y)

plt.figure(figsize=(10,6))
plot_tree(clf)
plt.savefig("decision_tree.png")
plt.close()

# ---------- A7 Decision Boundary (2 features) ----------
X2 = X_binned[:, :2]

clf2 = DecisionTreeClassifier(max_depth=3)
clf2.fit(X2, y)

xx, yy = np.meshgrid(
    np.linspace(X2[:,0].min(), X2[:,0].max(), 200),
    np.linspace(X2[:,1].min(), X2[:,1].max(), 200)
)

grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf2.predict(grid).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X2[:,0], X2[:,1], c=y)
plt.savefig("decision_boundary.png")
plt.close()