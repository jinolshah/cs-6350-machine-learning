import numpy as np
import scipy.optimize as opt
from load_data import load_data



def calc_cons(alpha, y):
    a = np.reshape(alpha,(1, -1))
    b = np.reshape(y, (-1,1))

    return np.matmul(a, b)[0]


def compute_gaussian_kernel(data1, data2, gamma):
    expanded_data1 = np.reshape(np.tile(data1, (1, data2.shape[0])), (-1, data1.shape[1]))
    expanded_data2 = np.tile(data2, (data1.shape[0], 1))
    kernel_matrix = np.exp(-np.sum((expanded_data1 - expanded_data2) ** 2, axis=1) / gamma)
    return kernel_matrix.reshape(data1.shape[0], data2.shape[0])


def gaussian_dual_objective(alpha, kernel_matrix, labels):
    term1 = -np.sum(alpha)
    alpha_labels = alpha[:, None] * labels[:, None]
    term2 = 0.5 * np.sum((alpha_labels @ alpha_labels.T) * kernel_matrix)
    return term1 + term2


def train_gaussian_svm(C, gamma, features, labels):
    n_samples = features.shape[0]
    kernel = compute_gaussian_kernel(features, features, gamma)
    constraints = {'type': 'eq', 'fun': lambda alpha: np.dot(alpha, labels)}
    result = opt.minimize(
        fun=lambda alpha: gaussian_dual_objective(alpha, kernel, labels),
        x0=np.zeros(n_samples),
        method='SLSQP',
        bounds=[(0, C)] * n_samples,
        constraints=constraints,
        options={'disp': False}
    )
    return result.x


def evaluate_gaussian_svm(gamma, alpha, train_features, train_labels, test_features, test_labels):
    kernel = compute_gaussian_kernel(train_features, test_features, gamma)
    weighted_kernel = train_labels[:, None] * kernel
    predictions = np.sum(alpha[:, None] * weighted_kernel, axis=0)
    predictions = np.where(predictions > 0, 1, -1)
    error_rate = np.sum(np.abs(predictions - test_labels)) / (2 * len(test_labels))
    return error_rate


if __name__ == "__main__":
    train_X, train_y, test_X, test_y = load_data()
    num_features = train_X.shape[1]
    
    C_vals = np.array([100, 500, 700]) / 873
    gamma_vals = np.array([0.1, 0.5, 1, 5, 100])

    print('')
    print("--/ Gaussian svm /--")

    overlap_index = 0
    for C in C_vals:
        for gamma in gamma_vals:
            alphas = train_gaussian_svm(C, gamma, train_X[:, :num_features - 1], train_y)
            support_vectors = np.where(alphas > 0)[0]
            
            print('')
            print("C: ", C)
            print("Support Vectors: ", len(support_vectors))

            train_error = evaluate_gaussian_svm(gamma, alphas, train_X[:, :num_features - 1], train_y,
                                                train_X[:, :num_features - 1], train_y)
            test_error = evaluate_gaussian_svm(gamma, alphas, train_X[:, :num_features - 1], train_y,
                                               test_X[:, :num_features - 1], test_y)

            print("Gamma: ", gamma)
            print("Train Error: ", train_error)
            print("Test Error: ", test_error)

            if(C == 500/873):
                if overlap_index > 0:
                    overlap_count = len(np.intersect1d(support_vectors, prev_sps))
                    print(f"Overlapping SVs between gamma: {gamma_vals[overlap_index]} "
                          f", {gamma_vals[overlap_index - 1]} = {overlap_count}")
                overlap_index += 1
                prev_sps = support_vectors