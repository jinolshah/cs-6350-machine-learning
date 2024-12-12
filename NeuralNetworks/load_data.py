import pandas as pd
import numpy as np
import os

def adjust_labels(y):
    y_copy = y.copy()
    y_copy[y_copy == 0] = -1
    return y_copy

def split_data(filepath):
    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, 0:4].values
    y = adjust_labels(df.iloc[:, 4].values)
    return X, y

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, "bank-note/train.csv")
    test_path = os.path.join(base_path, "bank-note/test.csv")

    train_X, train_y = split_data(train_path)
    test_X, test_y = split_data(test_path)

    return train_X, train_y, test_X, test_y
