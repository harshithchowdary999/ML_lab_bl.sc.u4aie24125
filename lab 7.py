import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

import shap
from lime.lime_tabular import LimeTabularExplainer


# ---------- Load Dataset ----------
data = pd.read_csv("cmu_mosi_numeric.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------- A2: Hyperparameter Tuning ----------
param_dist = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
svm = SVC(probability=True)

random_search = RandomizedSearchCV(svm, param_dist, n_iter=5, cv=3)
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

# ---------- A3: Classification Models ----------
models = {
    "SVM": best_model,
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(max_iter=500)
}

print("\n--- Classification Results ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred))


# ---------- A4: Regression ----------
print("\n--- Regression Results ---")

reg1 = LinearRegression()
reg1.fit(X_train, y_train)
y_pred = reg1.predict(X_test)

print("\nLinear Regression")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

reg2 = RandomForestRegressor()
reg2.fit(X_train, y_train)
y_pred = reg2.predict(X_test)

print("\nRandom Forest Regressor")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))


# ---------- A5: Clustering ----------
print("\n--- Clustering ---")

kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

print("KMeans Silhouette:", silhouette_score(X, labels))

db = DBSCAN()
labels_db = db.fit_predict(X)

if len(set(labels_db)) > 1:
    print("DBSCAN Silhouette:", silhouette_score(X, labels_db))
else:
    print("DBSCAN produced single cluster")


# ---------- O1: SHAP ----------
print("\n--- SHAP Feature Importance ---")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:50])

shap.summary_plot(shap_values, X_test[:50], show=False)
plt.savefig("shap_plot.png")
plt.close()


# ---------- O2: LIME ----------
print("\n--- LIME Explanation ---")

explainer_lime = LimeTabularExplainer(
    X_train,
    feature_names=[f"f{i}" for i in range(X.shape[1])],
    class_names=["0", "1"],
    discretize_continuous=True
)

exp = explainer_lime.explain_instance(
    X_test[0],
    rf.predict_proba
)

exp.save_to_file("lime_explanation.html")