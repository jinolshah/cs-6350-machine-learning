import getData
import numpy as np
import matplotlib.pyplot as plt
from forest import *

if __name__ == "__main__":
    
    train_data, test_data = getData.getData()

    numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for numeric_attr in numeric_attributes:
      median_thr(train_data, numeric_attr)
      median_thr(test_data, numeric_attr)
    
    train_errors = []
    test_errors=[]
    numberOfFeatures = [2, 4, 6]
    for num_features in numberOfFeatures:
        rf_train_error, rf_test_error = RandomForest(train_data, test_data, num_features, 500, 1000)
        plt.figure(figsize=(10, 6))
        iterations = list(range(500))
        plt.plot(iterations, rf_train_error, label='Train Errors', color='blue')
        plt.plot(iterations, rf_test_error, label='Test Errors', color='red')
        plt.title('Train and Test Errors vs. Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Train and Test Errors')
        plt.legend()
        plt.grid(True)
        plt.show()
