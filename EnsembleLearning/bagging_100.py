import getData
import numpy as np
from bag import *

if __name__ == "__main__":
    
    train_data, test_data = getData.getData()

    numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    for numeric_attr in numeric_attributes:
      median_thr(train_data, numeric_attr)
      median_thr(test_data, numeric_attr)
    
    train_errors = []
    iterations = 100
    numberOfTrees = 500
    test_errors=[]
    single_tree_test_errors = []
    for iter in range(iterations):
        bag_train_error, bag_test_error, single_tree_test_error = bagging_decision_trees(train_data, test_data, numberOfTrees, 1000)
        train_errors.append(bag_train_error)
        test_errors.append(bag_test_error)
        single_tree_test_errors.append(single_tree_test_error)

    value = np.array(test_data.iloc[:, -1].tolist())
    value[value == 'yes'] = 1
    value[value == 'no'] = -1
    print(value)
    print(single_tree_test_errors)
    value = value.astype(int)
    bias = np.mean(np.square(single_tree_test_errors - value))
    mean = np.mean(single_tree_test_errors) 
    variance = np.sum(np.square(single_tree_test_errors - mean)) / (len(test_data) - 1)
    print("bias =",bias)
    print("variance =",variance)
    print('bias + variance in 100 single tree predictor = ', bias+variance)
    for row in range(len(test_errors)):
      test_errors[row] = np.mean(test_errors[row])
    test_errors = np.sum(test_errors,axis=0) / (iterations * numberOfTrees)
    bias = np.mean(np.square(test_errors - value))
    mean = np.mean(test_errors)
    variance = np.sum(np.square(test_errors - mean)) / (len(test_data) - 1)
    print("bias =",bias)
    print("variance =",variance)
    print('bias + variance in 100 bagged tree predictor =', bias+variance)