import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score


def load_dataset(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y


def regression_one_feature(X, y):
    X1 = X.iloc[:, [0]]

    X_train, X_test, y_train, y_test = train_test_split(
        X1, y, test_size=0.3, random_state=42
    )

    reg = LinearRegression().fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred


def regression_all_features(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    reg = LinearRegression().fit(X_train, y_train)

    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)

    return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred


def regression_metrics(y_true, y_pred):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

    r2 = r2_score(y_true, y_pred)

    return mse, rmse, mape, r2


def perform_kmeans(X_train, k):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X_train)

    return kmeans


def clustering_scores(X_train, labels):

    sil = silhouette_score(X_train, labels)

    ch = calinski_harabasz_score(X_train, labels)

    db = davies_bouldin_score(X_train, labels)

    return sil, ch, db


def evaluate_k_values(X_train):

    sil_scores = []
    ch_scores = []
    db_scores = []

    for k in range(2, 10):

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_train)

        sil_scores.append(silhouette_score(X_train, kmeans.labels_))
        ch_scores.append(calinski_harabasz_score(X_train, kmeans.labels_))
        db_scores.append(davies_bouldin_score(X_train, kmeans.labels_))

    return sil_scores, ch_scores, db_scores


def elbow_method(X_train):

    distortions = []

    for k in range(2, 20):

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(X_train)

        distortions.append(kmeans.inertia_)

    return distortions


X, y = load_dataset("cmu_mosi_numeric.csv")


X_train1, X_test1, y_train1, y_test1, y_train_pred1, y_test_pred1 = regression_one_feature(X, y)

train_metrics1 = regression_metrics(y_train1, y_train_pred1)
test_metrics1 = regression_metrics(y_test1, y_test_pred1)

print("A1-A2 One Feature Train Metrics:", train_metrics1)
print("A1-A2 One Feature Test Metrics:", test_metrics1)


X_train_all, X_test_all, y_train_all, y_test_all, y_train_pred_all, y_test_pred_all = regression_all_features(X, y)

train_metrics_all = regression_metrics(y_train_all, y_train_pred_all)
test_metrics_all = regression_metrics(y_test_all, y_test_pred_all)

print("A3 Train Metrics:", train_metrics_all)
print("A3 Test Metrics:", test_metrics_all)


kmeans = perform_kmeans(X_train_all, 2)

sil, ch, db = clustering_scores(X_train_all, kmeans.labels_)

print("A5 Silhouette:", sil)
print("A5 CH Score:", ch)
print("A5 DB Index:", db)


sil_scores, ch_scores, db_scores = evaluate_k_values(X_train_all)

k_values = list(range(2, 10))


plt.plot(k_values, sil_scores)
plt.title("Silhouette Score vs K")
plt.savefig("silhouette_plot.png")
plt.close()


plt.plot(k_values, ch_scores)
plt.title("Calinski-Harabasz Score vs K")
plt.savefig("ch_score_plot.png")
plt.close()


plt.plot(k_values, db_scores)
plt.title("Davies-Bouldin Score vs K")
plt.savefig("db_index_plot.png")
plt.close()


distortions = elbow_method(X_train_all)


plt.plot(range(2, 20), distortions)
plt.title("Elbow Method")
plt.savefig("elbow_plot.png")
plt.close()