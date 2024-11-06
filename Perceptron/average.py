import numpy as np

def average_perceptron(X, y, num_epochs, learning_rate):
    samples, features = X.shape
    weights = np.zeros(features)
    averages = np.zeros(features)
    indices = np.arange(samples)
    for _ in range(num_epochs):
        np.random.shuffle(indices)
        X = X[indices, :]
        y = y[indices]
        for i in range(samples):
            predicted = np.sum(weights * X[i])
            if predicted * y[i] <= 0:
                weights += learning_rate * y[i] * X[i]
            averages = averages + weights
    return averages