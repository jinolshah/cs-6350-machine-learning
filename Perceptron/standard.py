import numpy as np

def standard_perceptron(X, y, num_epochs, learning_rate):
    samples, features = X.shape
    weights = np.zeros(features)
    for _ in range(num_epochs):
        for idx in np.random.permutation(samples):
            result = np.dot(weights, X[idx])
            if result * y[idx] <= 0:
                weights += learning_rate * y[idx] * X[idx]
    return weights

def standard_evaluate(X, y, weights):
    outputs = np.sign(np.dot(X, weights))
    outputs[outputs == 0] = -1
    misclassified = np.mean(outputs != y)
    return misclassified
