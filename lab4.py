import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

def load_dataset(path):
    data = pd.read_csv(path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y

def train_test_knn(X, y, k=3):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return cm, precision, recall, f1

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

def plot_scatter(X, y):
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c="blue")
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c="red")
    plt.savefig("scatter_plot.png")
    plt.close()

def plot_decision_boundary(X, y, k):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min(), X[:, 0].max(), 200),
        np.linspace(X[:, 1].min(), X[:, 1].max(), 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    Z = model.predict(grid).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.savefig(f"decision_boundary_k_{k}.png")
    plt.close()

def tune_k(X, y):
    params = {"n_neighbors": list(range(1, 15))}
    grid = GridSearchCV(KNeighborsClassifier(), params, cv=5)
    grid.fit(X, y)
    return grid.best_params_, grid.best_score_

X, y = load_dataset("cmu_mosi_numeric.csv")

X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_test_knn(X, y)

cm_train, p_tr, r_tr, f1_tr = classification_metrics(y_train, y_train_pred)
cm_test, p_te, r_te, f1_te = classification_metrics(y_test, y_test_pred)

print("Train Confusion Matrix:\n", cm_train)
print("Test Confusion Matrix:\n", cm_test)
print("Test Precision:", p_te)
print("Test Recall:", r_te)
print("Test F1:", f1_te)

mse, rmse, mape, r2 = regression_metrics(y_test, y_test_pred)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2:", r2)

X2 = X[:, :2]
plot_scatter(X2, y)

for k in [1, 3, 7]:
    plot_decision_boundary(X2, y, k)

best_k, best_score = tune_k(X_train, y_train)
print("Best k:", best_k)
print("Best CV Score:", best_score)
