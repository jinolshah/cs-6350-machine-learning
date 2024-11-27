import numpy as np
import scipy.optimize as opt
from load_data import load_data



def calc_cons(alpha, y):
    a = np.reshape(alpha,(1, -1))
    b = np.reshape(y, (-1,1))

    return np.matmul(a, b)[0]


def obj_func(alpha, X, y):
    function = -1 * np.sum(alpha)

    a = np.reshape(alpha,(-1,1))
    b = np.reshape(y, (-1,1))
    c = np.multiply(a, b)

    alpha_yX = np.multiply(c, X)
    mm = np.matmul(alpha_yX, np.transpose(alpha_yX))
    function += 0.5 * np.sum(mm)

    return function


def make_dsvm(C, X, y):
    constraints = ({'type': 'eq', 'fun': lambda alpha: calc_cons(alpha, y)})
    n_samples = X.shape[0]

    optimized = opt.minimize(
        constraints=constraints,
        fun=lambda alpha: obj_func(alpha, X, y),
        x0=np.zeros(n_samples), 
        bounds=[(0, C)] * n_samples, 
        method='SLSQP',  
        options={'disp': False}
    )

    a = np.reshape(optimized.x,(-1,1))
    b = np.reshape(y, (-1,1))
    c = np.multiply(np.multiply(a, b), X)
    weights = np.sum(c, axis=0)

    support_vectors = np.where((optimized.x > 0) & (optimized.x < C))
    
    bias =  np.mean(y[support_vectors] - np.matmul(X[support_vectors,:], np.reshape(weights, (-1,1))))

    weights = weights.tolist()
    weights.append(bias)
    weights = np.array(weights)

    return weights, bias


def calc_make_dsvm(X, y, weights):
    weights = np.reshape(weights, (5,1))

    prediction = np.matmul(X, weights)

    prediction[prediction <= 0] = -1
    prediction[prediction > 0] = 1

    count = np.sum(np.abs(prediction - np.reshape(y,(-1,1)))) / 2
    error = count / len(y)

    return error


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_data()
    num_columns = train_X.shape[1]
    
    C_vals = np.array([100, 500, 700]) / 873

    print('')
    print("--/ Dual svm /--")

    for C in C_vals:
        weights_final, bias_ds = make_dsvm(C, train_X[:, :num_columns - 1], train_y)

        training_error = calc_make_dsvm(train_X, train_y, weights_final)
        testing_error = calc_make_dsvm(test_X, test_y, weights_final)

        print('')
        print("C: ", C)
        print("Weights: ", weights_final)
        print("Bias: ", bias_ds)
        print("Training Error: ", training_error)
        print("Testing Error: ", testing_error)