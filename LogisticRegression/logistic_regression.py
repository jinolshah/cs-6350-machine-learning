import numpy as np
from load_data import load_data

def calc_MAP(X, y, var, learning_rate_init, decay_param):
    samples, features = X.shape
    weights = np.zeros((1, features))
    sample_indices = np.arange(samples)
    
    for epoch in range(100):
        np.random.shuffle(sample_indices)
        X, y = X[sample_indices, :], y[sample_indices]
        
        for sample_idx in range(samples):
            current_sample = X[sample_idx, :].reshape(1, -1)
            interaction_term = y[sample_idx] * np.dot(weights, current_sample.T)
            gradient = - (samples * y[sample_idx] * current_sample) / (1 + np.exp(interaction_term)) + weights / var
            learning_rate = learning_rate_init / (1 + (learning_rate_init / decay_param) * epoch)
            weights -= learning_rate * gradient
            
    return weights.T

def maximum_likelihood_estimation(X, y, lr_initial, decay_rate):
    samples, features = X.shape
    weights = np.zeros((1, features))
    sample_indices = np.arange(samples)
    
    for epoch in range(100):
        np.random.shuffle(sample_indices)
        X, y = X[sample_indices, :], y[sample_indices]
        
        for sample_idx in range(samples):
            activation = y[sample_idx] * np.dot(weights, X[sample_idx, :])
            gradient = - (samples * y[sample_idx] * X[sample_idx, :]) / (1 + np.exp(activation))
            learning_rate = lr_initial / (1 + (lr_initial / decay_rate) * epoch)
            weights -= learning_rate * gradient
            
    return weights.T

def evaluate_logistic_regression(X, y, weight_vector):
    weight_vector = weight_vector.reshape((5, 1))
    predictions = np.dot(X, weight_vector)
    predictions[predictions > 0] = 1
    predictions[predictions <= 0] = -1
    misclassified = np.abs(predictions - y.reshape(-1, 1))
    error_rate = np.sum(misclassified) / (2 * len(y))
    return error_rate

if __name__ == "__main__":
    # Load training and testing datasets
    train_X, train_y, test_X, test_y = load_data()

    variance_values = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
    
    print("Logistic Regression\n")
    
    print("MAP Estimation:")
    for variance in variance_values:
        print()
        print("Variance:", variance)
        map_weights = calc_MAP(train_X, train_y, variance, 0.001, 0.1)
        train_error = evaluate_logistic_regression(train_X, train_y, map_weights)
        test_error = evaluate_logistic_regression(test_X, test_y, map_weights)
        print("Training Error:", train_error)
        print("Testing Error:", test_error)
    
    print()
    print("Maximum Likelihood Estimation:")
    mle_weights = maximum_likelihood_estimation(train_X, train_y, 0.001, 0.1)
    train_error = evaluate_logistic_regression(train_X, train_y, mle_weights)
    test_error = evaluate_logistic_regression(test_X, test_y, mle_weights)
    print("Training Error:", train_error)
    print("Testing Error:", test_error)
