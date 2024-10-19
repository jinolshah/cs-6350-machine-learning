import numpy as np
import pandas as pd
import dataset
import matplotlib.pyplot as plt
import os

def cost_function(data, weight_vector):
    sum = 0.0
    m = data.shape[0]
    for i in range(m):
        sum += (data.iloc[i][-1] - np.dot(weight_vector, data.iloc[i][:-1]))**2
    return sum/2

def batch_gradient_cost(data, weight_vector):
    m = data.shape[0]
    gradient_vector=np.asarray([0.0] * (data.shape[1] -1))
    for columns in range(data.shape[1] -1) :
        sum = 0.0
        for i in range(m):
            sum+=(data.iloc[i][-1] - np.dot(weight_vector, data.iloc[i][:-1])) * data.iloc[i][columns]
        gradient_vector[columns] = -sum/m
    return gradient_vector

def update_weight_vector(weight_vector, learning_rate, gradient_vector):
    return np.asarray(weight_vector) - learning_rate*np.asarray(gradient_vector)

def stochastic_gradient_cost(row, data, weight_vector, learning_rate):
    new_weight_vector=np.asarray([0.0] * len(weight_vector))
    for columns in range(len(weight_vector)):
        new_weight_vector[columns] = weight_vector[columns] + learning_rate*((data.iloc[row][-1] - np.dot(weight_vector, data.iloc[row][:-1]))*data.iloc[row][columns])
    return new_weight_vector

def GradientDescent(df_train, df_test, weight_vector, learning_rate, max_iterations):
    cost_iterations_list = []
    weight_vector_list = []
    iteration = 1
    while True:
        if iteration<max_iterations:
            for row in range(df_train.shape[0]):
                new_weight_vector = stochastic_gradient_cost(row, df_train, weight_vector, learning_rate)
            cost_iterations_list.append(cost_function(df_train, weight_vector))
            if np.linalg.norm(new_weight_vector - weight_vector) < 0.000001:
                break
            else:
                weight_vector = new_weight_vector
                iteration+=1
                weight_vector_list.append(weight_vector)
        else:
            learning_rate /= 4
            weight_vector = np.asarray([0.0] * (df_train.shape[1] - 1))
            weight_vector_list = list()
            cost_iterations_list = list()
            weight_vector_list.append(weight_vector)
            iteration = 1
        
    print("The final weight vector is: ", weight_vector_list[-1])
    print("The learning rate is: ", learning_rate)
    print("Cost of test data set: ", cost_function(df_test, weight_vector_list[-1]))
    plt.plot(range(len(cost_iterations_list)), cost_iterations_list)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.title('Cost Function over Iterations')
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bank_train_data_file = os.path.join(script_dir, "concrete/train.csv")
    bank_test_data_file = os.path.join(script_dir, "concrete/train.csv")

    df_train = pd.read_csv(bank_train_data_file)
    df_train.columns = dataset.concrete_columns
    df_test = pd.read_csv(bank_test_data_file)
    df_test.columns = dataset.concrete_columns

    weight_vector = np.asarray([0.0] * (df_train.shape[1] - 1))

    GradientDescent(df_train, df_test, weight_vector,  1, 5000)