import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


# ---------- A1 FUNCTIONS ----------
def summation(x, w):
    return np.dot(x, w)

def step(x):
    return 1 if x >= 0 else 0

def bipolar_step(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    x = np.clip(x, -500, 500)   # FIXED overflow
    return 1 / (1 + np.exp(-x))

def relu(x):
    return max(0, x)

def error(t, y):
    return t - y


# ---------- FIXED PERCEPTRON ----------
def train_perceptron(X, y, activation, lr=0.05, max_epochs=1000):
    w = np.random.rand(X.shape[1] + 1)
    errors = []

    for epoch in range(max_epochs):
        sse = 0
        for i in range(len(X)):
            x = np.insert(X[i], 0, 1)
            y_pred = activation(summation(x, w))
            e = error(y[i], y_pred)
            w = w + lr * e * x
            sse += e**2

        errors.append(sse)
        if sse <= 0.002:
            break

    return w, errors


# ---------- DATA ----------
def AND_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,0,0,1])
    return X, y

def XOR_data():
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([0,1,1,0])
    return X, y

def customer_data():
    data = np.array([
        [20,6,2,386,1],
        [16,3,6,289,1],
        [27,6,2,393,1],
        [19,1,2,110,0],
        [24,4,2,280,1],
        [22,1,5,167,0],
        [15,4,2,271,1],
        [18,4,2,274,1],
        [21,1,4,148,0],
        [16,2,4,198,0]
    ])
    return data[:,:-1], data[:,-1]


# ---------- PLOT ----------
def plot_error(errors, name):
    plt.plot(errors)
    plt.title(name)
    plt.savefig(name + ".png")
    plt.close()


# ---------- PSEUDO-INVERSE ----------
def pseudo_inverse(X, y):
    X = np.c_[np.ones(len(X)), X]
    return np.linalg.pinv(X) @ y


# ---------- BACKPROP ----------
def backprop_AND():
    X, y = AND_data()
    np.random.seed(0)

    W1 = np.random.rand(2,2)
    W2 = np.random.rand(2,1)

    lr = 0.05
    errors = []

    for epoch in range(1000):
        total_error = 0

        for i in range(len(X)):
            x = X[i]
            target = y[i]

            h = sigmoid(np.dot(x, W1))
            o = sigmoid(np.dot(h, W2))

            e = target - o
            total_error += e**2

            d_o = e * o * (1 - o)
            d_h = h * (1 - h) * (d_o @ W2.T)

            W2 += lr * np.outer(h, d_o)
            W1 += lr * np.outer(x, d_h)

        errors.append(total_error)
        if total_error <= 0.002:
            break

    return errors


# ---------- TWO OUTPUT ----------
def two_output_encoding(y):
    return np.array([[1,0] if i==0 else [0,1] for i in y])


# ---------- IMPROVED MLP ----------
def mlp_model(X, y):
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000, learning_rate_init=0.01)
    clf.fit(X, y)
    return clf


# ---------- MAIN ----------
if __name__ == "__main__":

    # A2 AND
    X, y = AND_data()
    _, err = train_perceptron(X, y, step)
    plot_error(err, "AND_step")

    # A3 activations
    for func, name in [(bipolar_step,"bipolar"), (sigmoid,"sigmoid"), (relu,"relu")]:
        _, err = train_perceptron(X, y, func)
        plot_error(err, "AND_"+name)

    # A4 learning rate
    rates = np.arange(0.1,1.1,0.1)
    iterations = []

    for r in rates:
        _, err = train_perceptron(X, y, step, lr=r)
        iterations.append(len(err))

    plt.plot(rates, iterations)
    plt.savefig("learning_rate.png")
    plt.close()

    # A5 XOR
    X, y = XOR_data()
    _, err = train_perceptron(X, y, step)
    plot_error(err, "XOR")

    # A6 customer
    Xc, yc = customer_data()
    _, err = train_perceptron(Xc, yc, sigmoid)
    plot_error(err, "customer")

    # A7 pseudo inverse
    w = pseudo_inverse(Xc, yc)

    # A8 backprop
    err = backprop_AND()
    plot_error(err, "backprop")

    # A10 encoding
    y_encoded = two_output_encoding(y)

    # A11 AND MLP
    X, y = AND_data()
    clf = mlp_model(X, y)

    # A12 project dataset
    try:
        data = pd.read_csv("cmu_mosi_numeric.csv")
        Xp = data.iloc[:,:-1].values
        yp = data.iloc[:,-1].values
        clf = mlp_model(Xp, yp)
    except:
        pass