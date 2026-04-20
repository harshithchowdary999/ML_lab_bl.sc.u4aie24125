import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import shap
from lime.lime_tabular import LimeTabularExplainer


# ---------- LOAD DATA ----------
data = pd.read_csv("cmu_mosi_numeric.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ---------- A1: CORRELATION ----------
def correlation_analysis(X):
    corr = X.corr()

    plt.figure(figsize=(10,8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("correlation_heatmap.png")
    plt.close()

correlation_analysis(X)


# ---------- SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------- MODEL ----------
def run_model(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), model


# ---------- A2: PCA 99% ----------
def pca_experiment(var):
    pca = PCA(n_components=var)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    acc, model = run_model(X_train_pca, X_test_pca, y_train, y_test)

    return acc, pca, model, X_train_pca, X_test_pca


acc_99, pca_99, model_99, X_train_99, X_test_99 = pca_experiment(0.99)
print("Accuracy with 99% variance:", acc_99)


# ---------- A3: PCA 95% ----------
acc_95, pca_95, model_95, X_train_95, X_test_95 = pca_experiment(0.95)
print("Accuracy with 95% variance:", acc_95)


# ---------- ORIGINAL ----------
acc_original, base_model = run_model(X_train, X_test, y_train, y_test)
print("Original Accuracy:", acc_original)


# ---------- A4: SEQUENTIAL FEATURE SELECTION ----------
def sfs_experiment():
    model = RandomForestClassifier()

    sfs = SequentialFeatureSelector(
        model,
        n_features_to_select=10,
        direction='forward'
    )

    sfs.fit(X_train, y_train)

    X_train_sfs = sfs.transform(X_train)
    X_test_sfs = sfs.transform(X_test)

    acc, _ = run_model(X_train_sfs, X_test_sfs, y_train, y_test)

    return acc


acc_sfs = sfs_experiment()
print("Accuracy with SFS:", acc_sfs)


# ---------- A5: SHAP ----------
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_test[:50])

shap.summary_plot(shap_values, X_test[:50], show=False)
plt.savefig("shap_plot.png")
plt.close()


# ---------- A5: LIME ----------
explainer_lime = LimeTabularExplainer(
    X_train,
    feature_names=[f"f{i}" for i in range(X.shape[1])],
    class_names=["0", "1"],
    discretize_continuous=True
)

exp = explainer_lime.explain_instance(
    X_test[0],
    base_model.predict_proba
)

exp.save_to_file("lime_explanation.html")


print("\nAll tasks completed successfully ✅")