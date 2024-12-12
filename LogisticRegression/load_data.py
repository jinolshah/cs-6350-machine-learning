import pandas as pd
import numpy as np
import os

def split_data(filepath):
    df = pd.read_csv(filepath, header=None)
    data_values = df.values
    num_features = data_values.shape[1]
    X = np.copy(data_values)
    X[:, num_features - 1] = 1
    y = data_values[:, num_features - 1]
    y = 2 * y - 1
    return X, y

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "bank-note/train.csv")
    test_path = os.path.join(base_path, "bank-note/test.csv")
    
    train_X, train_y = split_data(train_path)
    test_X, test_y = split_data(test_path)

    return train_X, train_y, test_X, test_y
