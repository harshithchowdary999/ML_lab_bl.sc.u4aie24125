import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from lime.lime_tabular import LimeTabularExplainer


# ---------- LOAD DATA ----------
data = pd.read_csv("cmu_mosi_numeric.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ---------- A1: STACKING ----------
def stacking_model():
    base_models = [
        ('dt', DecisionTreeClassifier()),
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True)),
        ('nb', GaussianNB())
    ]

    final_model = LogisticRegression(max_iter=1000)

    stack = StackingClassifier(
        estimators=base_models,
        final_estimator=final_model
    )

    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)

    return stack, y_pred


# ---------- A2: PIPELINE ----------
def pipeline_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return pipeline, y_pred


# ---------- METRICS ----------
def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)


# ---------- A3: LIME ----------
def lime_explanation(model):
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=[f"f{i}" for i in range(X.shape[1])],
        class_names=["0", "1"],
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        X_test[0],
        model.predict_proba
    )

    exp.save_to_file("lime_pipeline.html")
    print("LIME explanation saved as lime_pipeline.html")


# ---------- MAIN ----------
if __name__ == "__main__":

    print("\n--- A1: Stacking ---")
    stack_model, y_pred_stack = stacking_model()
    evaluate(y_test, y_pred_stack)

    print("\n--- A2: Pipeline ---")
    pipe_model, y_pred_pipe = pipeline_model()
    evaluate(y_test, y_pred_pipe)

    print("\n--- A3: LIME ---")
    lime_explanation(pipe_model)

    print("\nProgram completed successfully ✅")