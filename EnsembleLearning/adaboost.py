import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import dataset
import os

#replace numeric attribute values with median threshold
def median_thresholding(df, attribute):
  threshold = df[attribute].median()
  df[attribute] = (df[attribute] >= threshold).astype(int)

script_dir = os.path.dirname(os.path.abspath(__file__))
bank_train_data_file = os.path.join(script_dir, "bank-4/train.csv")
bank_test_data_file = os.path.join(script_dir, "bank-4/train.csv")

df_train = pd.read_csv(bank_train_data_file)
df_train.columns = dataset.bank_columns
df_test = pd.read_csv(bank_test_data_file)
df_test.columns = dataset.bank_columns
numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
for numeric_attr in numeric_attributes:
    median_thresholding(df_train, numeric_attr)
    median_thresholding(df_test, numeric_attr)

X_train = df_train.drop(df_train.columns[-1], axis=1).values

X_train = X_train[:]

y_train = df_train.iloc[:,-1].apply(lambda x: 1 if x == 'yes' else -1).values

X_test = df_test.drop(df_train.columns[-1], axis=1).values

X_test = X_test[:]

y_test = df_test.iloc[:,-1].apply(lambda x: 1 if x == 'yes' else -1).values


class DecisionTreeStump:
    def __init__(self, attribute, threshold, label_lesser, label_greater):
        self.attribute = attribute
        self.threshold = threshold
        self.label_lesser = label_lesser
        self.label_greater = label_greater

    def predict(self, x):
        if x[self.attribute] <= self.threshold:
            return self.label_lesser
        else:
            return self.label_greater

    @staticmethod
    def weighted_information_gain(X, y, attribute, threshold, weights):
        indices_less_threshold = X[:, attribute] <= threshold
        indices_greater_threshold = X[:, attribute] > threshold

        weighted_less_indices = np.sum(weights[indices_less_threshold])
        weighted_greater_indices = np.sum(weights[indices_greater_threshold])

        if weighted_less_indices == 0 or weighted_greater_indices == 0:
            return 0

        weighted_entropy_less = -np.sum(weights[indices_less_threshold] * np.log2(weights[indices_less_threshold] / weighted_less_indices))
        weighted_greater_entropy = -np.sum(weights[indices_greater_threshold] * np.log2(weights[indices_greater_threshold] / weighted_greater_indices))

        total_entropy = (weighted_less_indices / len(y)) * weighted_entropy_less + (weighted_greater_indices / len(y)) * weighted_greater_entropy
        return total_entropy

    @staticmethod
    def split_max_IG(X, y, weights):
        numberOfFeatures = X.shape[1]
        selected_threshold = None
        selected_attribute = None
        min_entropy = float('inf')

        for features in range(numberOfFeatures):
            unique_values = np.unique(X[:, features])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                entropy = DecisionTreeStump.weighted_information_gain(X, y, features, threshold, weights)

                if entropy < min_entropy:
                    min_entropy = entropy
                    selected_threshold = threshold
                    selected_attribute = features

        return selected_attribute, selected_threshold


class AdaBoost:
    def __init__(self, iterations, learningRate=0.5):
        self.iterations = iterations
        self.learningRate = learningRate
        self.alpha = []
        self.stumps = []
        self.train_errors = []
        self.test_errors = []
        self.stump_errors = []

    def fit(self, X, y):
        X = self.numeric_encoding(X)
        y = y.astype(float)
        weights = np.ones(len(y)) / len(y)

        for t in range(self.iterations):
            stump = self.train_stump(X, y, weights)
            self.stumps.append(stump)
            predictions = np.array([stump.predict(x) for x in X]).astype(float)
            #calculate error
            error = np.sum(weights * (predictions != y))

            # calculate new alpha and update it
            if error == 0:
                alpha = 1.0
            else:
                alpha = self.learningRate * np.log((1 - error) / max(error, 1e-10))
            self.alpha.append(alpha)

            #adjust weights based on calculated alpha
            weights = weights * np.exp(-alpha * y * predictions)
            weights = weights / np.sum(weights)

            train_error = 1 - accuracy_score(y_train, self.predict(X_train))
            test_error = 1 - accuracy_score(y_test, self.predict(X_test))
            self.train_errors.append(train_error)
            self.test_errors.append(test_error)
            self.stump_errors.append(error / len(y))

    def numeric_encoding(self, X):
        for i in range(X.shape[1]):
            columns = X[:, i]
            if not np.issubdtype(columns.dtype, np.number):
                columns = pd.to_numeric(columns, errors='coerce')
                columns[np.isnan(columns)] = 0
            X[:, i] = columns
        return X

    def train_stump(self, X, y, weights):
        best_error = float('inf')
        selected_stump = None
        for _ in range(2):
            attribute, threshold = DecisionTreeStump.split_max_IG(X, y, weights)
            for label_lesser in [-1, 1]:
                predictions = np.where(X[:, attribute] <= threshold, label_lesser, -label_lesser)
                error = np.sum(weights * (predictions != y))
                if best_error >= error:
                    best_error = error
                    selected_stump = DecisionTreeStump(attribute, threshold, label_lesser, -label_lesser)
        return selected_stump

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alpha, self.stumps):
            predictions += alpha * np.array([stump.predict(x) for x in X])
        return np.sign(predictions)

iterations = 500
learningRate = 0.3
boosting = AdaBoost(iterations, learningRate)

boosting.fit(X_train, y_train)

train_error = 1 - accuracy_score(y_train, boosting.predict(X_train))
test_error = 1 - accuracy_score(y_test, boosting.predict(X_test))

print("Training Error = ", train_error)
print("Testing Error = ", test_error)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(iterations), boosting.train_errors, label='Train Error')
plt.plot(range(iterations), boosting.test_errors, label='Test Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Train and Test Errors vs. Iteration')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(iterations), boosting.stump_errors, label='Decision Stump Error')
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Decision Stump Errors vs. Iteration')
plt.legend()

plt.tight_layout()
plt.show()