import numpy as np
from load_data import load_data

def compute_kernel_perceptron(features, labels, gamma):
    num_samples = features.shape[0]
    indices = np.arange(num_samples)
    coefficients = np.zeros((num_samples, 1))
    labels = labels[:, None]
    kernel_matrix = compute_gaussian_kernel(features, features, gamma)
    
    for _ in range(100):
        np.random.shuffle(indices)
        for idx in indices:
            weighted_sum = np.sum(coefficients * labels * kernel_matrix[idx, :])
            if weighted_sum * labels[idx] <= 0:
                coefficients[idx] += 1
    return coefficients


def evaluate_kernel_perceptron(coefficients, train_features, train_labels, test_features, test_labels, gamma):
    kernel_matrix = compute_gaussian_kernel(train_features, test_features, gamma)
    weighted_predictions = np.sum(coefficients * train_labels[:, None] * kernel_matrix, axis=0)
    predicted_labels = np.where(weighted_predictions > 0, 1, -1)
    error_rate = np.sum(predicted_labels != test_labels) / len(test_labels)
    return error_rate


def compute_gaussian_kernel(data1, data2, gamma):
    expanded_data1 = np.tile(data1, (1, data2.shape[0]))
    expanded_data1 = expanded_data1.reshape(-1, data1.shape[1])
    expanded_data2 = np.tile(data2, (data1.shape[0], 1))
    kernel = np.exp(-np.sum((expanded_data1 - expanded_data2) ** 2, axis=1) / gamma)
    return kernel.reshape(data1.shape[0], data2.shape[0])


if __name__ == "__main__":
    train_features, train_labels, test_features, test_labels = load_data()
    gamma_values = np.array([0.1, 0.5, 1, 5, 100])

    print('')
    print("--/ Kernel perceptron /--")

    for gamma in gamma_values:

        coefficients = compute_kernel_perceptron(train_features, train_labels, gamma)
        train_error = evaluate_kernel_perceptron(coefficients, train_features, train_labels, train_features, train_labels, gamma)
        test_error = evaluate_kernel_perceptron(coefficients, train_features, train_labels, test_features, test_labels, gamma)

        print('')
        print("Gamma: ", gamma)
        print("Train Error: ", train_error)
        print("Test Error: ", test_error)
