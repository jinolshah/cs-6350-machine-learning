import numpy as np

def voted_perceptron(X, y, num_epochs, learning_rate):
    samples, features = X.shape
    weights = np.zeros(features)
    weight_history = []
    vote_history = []
    vote_count = 0
    
    for _ in range(num_epochs):
        indices = np.random.permutation(samples)
        for i in indices:
            prediction = np.dot(weights, X[i])
            if prediction * y[i] <= 0:
                weight_history.append(weights.copy())
                vote_history.append(vote_count)
                weights += learning_rate * y[i] * X[i]
                vote_count = 1
            else:
                vote_count += 1

    weight_history = np.array(weight_history)
    vote_history = np.array(vote_history)
    return weight_history, vote_history

def voted_evaluate(X, y, weight_history, vote_history):
    vote_history = vote_history.reshape(-1, 1)
    predictions = np.dot(X, weight_history.T)
    predictions = np.where(predictions > 0, 1, -1)
    combined_predictions = np.dot(predictions, vote_history)
    combined_predictions = np.where(combined_predictions > 0, 1, -1)

    error_rate = np.mean(combined_predictions.flatten() != y)
    return error_rate
