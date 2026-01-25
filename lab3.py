import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import minkowski

def evaluate_vectors(A, B):
    dot_ans = 0
    for i in range(len(A)):
        dot_ans += A[i] * B[i]

    norm_A = 0
    norm_B = 0
    for i in range(len(A)):
        norm_A += A[i] ** 2
        norm_B += B[i] ** 2

    norm_A = math.sqrt(norm_A)
    norm_B = math.sqrt(norm_B)

    return dot_ans, norm_A, norm_B, np.dot(A, B), np.linalg.norm(A), np.linalg.norm(B)

def mean(data):
    return sum(data) / len(data)

def variance(data):
    mu = mean(data)
    return sum((x - mu) ** 2 for x in data) / len(data)

def std_dev(data):
    return math.sqrt(variance(data))

def dataset_stats(X):
    means = []
    stds = []
    for i in range(X.shape[1]):
        col = X[:, i]
        means.append(mean(col))
        stds.append(std_dev(col))
    return np.array(means), np.array(stds)

def feature_histogram(feature):
    hist, bins = np.histogram(feature, bins=10)
    return hist, bins, mean(feature), variance(feature)

def minkowski_distance(A, B, p):
    total = 0
    for i in range(len(A)):
        total += abs(A[i] - B[i]) ** p
    return total ** (1 / p)

def custom_knn_predict(X_train, y_train, test_vec, k):
    distances = []
    for i in range(len(X_train)):
        d = minkowski_distance(X_train[i], test_vec, 2)
        distances.append((d, y_train[i]))
    distances.sort(key=lambda x: x[0])

    neighbors = distances[:k]
    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1

    return max(votes, key=votes.get)

def custom_knn_accuracy(X_train, X_test, y_train, y_test, k):
    correct = 0
    for i in range(len(X_test)):
        pred = custom_knn_predict(X_train, y_train, X_test[i], k)
        if pred == y_test[i]:
            correct += 1
    return correct / len(y_test)

def confusion_matrix_manual(y_true, y_pred):
    TP = FP = FN = TN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        else:
            TN += 1
    return TP, FP, FN, TN

def performance_metrics(TP, FP, FN, TN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    return accuracy, precision, recall, f1

def main():
    data = pd.read_csv("cmu_mosi_numeric.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    A = X[0]
    B = X[1]

    print(evaluate_vectors(A, B))

    X0 = X[y == 0]
    X1 = X[y == 1]

    mean0, std0 = dataset_stats(X0)
    mean1, std1 = dataset_stats(X1)

    print(np.linalg.norm(mean0 - mean1))

    hist, bins, mu, var = feature_histogram(X[:, 0])
    print(mu, var)

    plt.hist(X[:, 0], bins=10)
    plt.show()

    dist = []
    for p in range(1, 11):
        dist.append(minkowski_distance(A, B, p))

    plt.plot(range(1, 11), dist)
    plt.show()

    print(minkowski(A, B, 2))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    print(neigh.score(X_test, y_test))

    y_pred = neigh.predict(X_test)

    acc = []
    for k in range(1, 12):
        acc.append(custom_knn_accuracy(X_train, X_test, y_train, y_test, k))

    plt.plot(range(1, 12), acc)
    plt.show()

    TP, FP, FN, TN = confusion_matrix_manual(y_test, y_pred)
    print(performance_metrics(TP, FP, FN, TN))

    w = np.linalg.pinv(X_train) @ y_train
    preds = np.dot(X_test, w)
    preds = np.where(preds >= 0.5, 1, 0)
    print(np.mean(preds == y_test))

main()
