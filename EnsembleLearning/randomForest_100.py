import getData
import numpy as np
from forest import *

if __name__ == "__main__":

    train_data, test_data = getData.getData()

    numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for numeric_attr in numeric_attributes:
      median_thr(train_data, numeric_attr)
      median_thr(test_data, numeric_attr)
    
    train_errors = []
    
    numberOfFeatures = [2, 4, 6]
    iterations = 10
    numberOfTrees = 20
    for num_features in numberOfFeatures:
        test_errors=[]
        single_tree_test_errors = []
        for iter in range(iterations):
            rf_train_error, rf_test_error, single_tree_test_error = RandomForest(train_data, test_data, num_features, numberOfTrees, 1000)
            test_errors.append(rf_test_error)
            single_tree_test_errors.append(single_tree_test_error)
    
        value = np.array(test_data.iloc[:iterations, -1].tolist())
        value[value == 'yes'] = 1
        value[value == 'no'] = -1
        value = value.astype(int)
        bias = np.mean(np.square(single_tree_test_errors - value))
        mean = np.mean(single_tree_test_errors) 
        variance = np.sum(np.square(single_tree_test_errors - mean)) / (iterations)
        print("bias =",bias)
        print("variance =",variance)
        print('bias + variance in 100 single tree predictor= ', bias+variance)
        for row in range(len(test_errors)):
            test_errors[row] = np.mean(test_errors[row])
        test_errors = np.asarray(test_errors)
        test_errors = np.sum(test_errors,axis=0) / (iterations * numberOfTrees)
        bias = np.mean(np.square(test_errors - value))
        mean = np.mean(test_errors)
        variance = np.sum(np.square(test_errors - mean)) / (iterations)
        print("bias =",bias)
        print("variance =",variance)
        print('bias + variance in 100 bagged tree predictor =', bias+variance)