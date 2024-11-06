import pandas as pd
import numpy as np
import os
from standard import *
from voted import *
from average import *

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    data_values = df.values
    num_features = data_values.shape[1]
    X = np.copy(data_values)
    X[:, num_features - 1] = 1
    y = data_values[:, num_features - 1]
    y = 2 * y - 1
    return X, y

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "bank-note/train.csv")
    test_path = os.path.join(base_path, "bank-note/test.csv")
    
    train_X, train_y = load_data(train_path)
    test_X, test_y = load_data(test_path)
    
    print('-'*50)
    # Standard Perceptron
    print("Standard Perceptron:")
    std_weights = standard_perceptron(train_X, train_y, num_epochs=10, learning_rate=0.01)
    std_error = standard_evaluate(test_X, test_y, std_weights)
    print("Weights:\n", std_weights)
    print("Error:", std_error)
    print('-'*50)
    
    # Voted Perceptron
    print("Voted Perceptron:")
    voted_weights, vote_counts = voted_perceptron(train_X, train_y, num_epochs=10, learning_rate=0.01)
    voted_error = voted_evaluate(test_X, test_y, voted_weights, vote_counts)
    print("Weights:\n", voted_weights)
    print("Vote counts:\n", vote_counts)
    print("Error:", voted_error)
    print('-'*50)

    # Average Perceptron
    print("Average Perceptron:")
    avg_weights = average_perceptron(train_X, train_y, num_epochs=10, learning_rate=0.01)
    avg_error = standard_evaluate(test_X, test_y, avg_weights)
    print("Weights:\n", avg_weights)
    print("Error:", avg_error)
    print('-'*50)