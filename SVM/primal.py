import numpy as np
from load_data import load_data



calc_lr1 = lambda a, T, learning_rate: learning_rate / (1 + (learning_rate / a) * T)
calc_lr2 = lambda _, T, learning_rate: learning_rate / (1 + T)


def primal_svm(learning_rate, C, X, y, calc_lr, a = 0):
    num_samples, num_features = X.shape
    weight_vector = np.zeros(num_features)
    indices = np.arange(num_samples)
    for T in range(100):
        np.random.shuffle(indices)
        X = X[indices,:]
        y = y[indices]
        for sample in range(num_samples):
            condition = y[sample] * np.sum(np.multiply(weight_vector, X[sample,:]))
            temp_weights = np.copy(weight_vector)
            temp_weights[num_features-1] = 0
            if condition <= 1:
                temp_weights = temp_weights - C * num_samples * y[sample] * X[sample,:]
            learning_rate = calc_lr(a, T, learning_rate)
            weight_vector = weight_vector - learning_rate * temp_weights

    return weight_vector, learning_rate


def primal_svm_evaluate(X, y, weights):
    prediction = np.matmul(X, weights)
    prediction[prediction>0] = 1
    prediction[prediction<=0] = -1
    error = np.sum(np.abs(prediction - np.reshape(y,(-1,1)))) / 2 / len(y)
    return error


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_data()
    num_columns = train_X.shape[1]
    
    C_list = np.array([100, 500, 700]) / 873

    for C in C_list:
        print('------------------------------')
        print(f'C: {C}')

        weights_lr1, final_lr1 = primal_svm(0.1, C, train_X, train_y, calc_lr1, a=0.1)
        weights_lr1 = weights_lr1[:, None]

        print('')
        print("Primal SVM LR - 1")
        print('')
        print(f"Weights:\n{weights_lr1}")
        print(f"Final Learning Rate: {final_lr1}")
        train_error_lr1 = primal_svm_evaluate(train_X, train_y, weights_lr1)
        print(f"Train Error: {train_error_lr1}")
        test_error_lr1 = primal_svm_evaluate(test_X, test_y, weights_lr1)
        print(f"Test Error: {test_error_lr1}")

        weights_lr2, final_lr2 = primal_svm(0.1, C, train_X, train_y, calc_lr2)
        weights_lr2 = weights_lr2[:, None] 

        print('')
        print("Primal SVM LR - 2")
        print('')
        print(f"Weights:\n{weights_lr2}")
        print(f"Final Learning Rate: {final_lr2}")
        train_error_lr2 = primal_svm_evaluate(train_X, train_y, weights_lr2)
        print(f"Train Error: {train_error_lr2}")
        test_error_lr2 = primal_svm_evaluate(test_X, test_y, weights_lr2)
        print(f"Test Error: {test_error_lr2}")