import dataset
import pandas as pd
import numpy as np
import os

def cost_function(data, weight_vector):
    sum = 0
    m = data.shape[0]
    for i in range(m):
        sum += (data.iloc[i][-1] - np.dot(weight_vector, data.iloc[i][:-1]))**2
    return sum/2

script_dir = os.path.dirname(os.path.abspath(__file__))
bank_train_data_file = os.path.join(script_dir, "concrete/train.csv")
bank_test_data_file = os.path.join(script_dir, "concrete/train.csv")

df_train = pd.read_csv(bank_train_data_file)
df_train.columns = dataset.concrete_columns
df_test = pd.read_csv(bank_test_data_file)
df_test.columns = dataset.concrete_columns

weight_vector = np.asarray([0.0] * (df_train.shape[1] - 1))
X = df_train.iloc[:, :-1].values
y = df_train.iloc[:, -1].values
X_transpose = X.T
XTX = X_transpose.dot(X)
XTX_inv = np.linalg.inv(XTX)
optimal_weight_vector = XTX_inv.dot(X_transpose).dot(y)

print("Cost of train data setusing optimal weight vector: ", cost_function(df_train, optimal_weight_vector))
print("Cost of train data set using SGD: ", cost_function(df_train,  [0.0204519,
-0.04324712, -0.01256854, 0.01299676, 0.00549673, -0.00340634, 0.03144457]))
print("Cost of train data set using BGD: ", cost_function(df_train,  [0.00240026, -
0.00307263, -0.00376702, 0.00759447, -0.00098704, -0.00114946, 0.00087231]))
print("Cost of test data setusing optimal weight vector: ", cost_function(df_test, optimal_weight_vector))
print("Cost of test data set using SGD: ", cost_function(df_test,  [0.0204519,
-0.04324712, -0.01256854, 0.01299676, 0.00549673, -0.00340634, 0.03144457]))
print("Cost of test data set using BGD: ", cost_function(df_test,  [0.00240026, -
0.00307263, -0.00376702, 0.00759447, -0.00098704, -0.00114946, 0.00087231]))
print(optimal_weight_vector)