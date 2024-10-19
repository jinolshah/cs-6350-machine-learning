import pandas as pd
import dataset
import numpy as np
import os

import pandas as pd
import numpy as np

def calc_entropy(df):
  attribute = df.keys()[-1]
  values = df[attribute].unique()
  entropy = 0.0
  for value in values:
    prob = df[attribute].value_counts()[value] / len(df[attribute])
    entropy += -prob * np.log2(prob)
  return entropy

def max_entropy_attribute(df, attributes):
  avg_entropy = float('inf')
  selected_attribute = None
  np.random.shuffle(attributes)
  for attribute in attributes:
    values = df[attribute].unique()
    entropy = 0.0
    for value in values:
      subset = df[df[attribute] == value]
      entropy += len(subset) / len(df) * calc_entropy(subset)
    if entropy < avg_entropy:
      avg_entropy = entropy
      selected_attribute = attribute
  return selected_attribute

def DecisionTreeClassifier(df, attributes=None,maxDepth = 17):
  if attributes is None:
    attributes = df.keys()[:-1].tolist()
  if len(df[df.keys()[-1]].unique()) == 1:
    return df[df.keys()[-1]].iloc[0]
  if len(attributes) == 0:
    return df[df.keys()[-1]].value_counts().idxmax()
  
  selected_attribute = max_entropy_attribute(df, attributes)
  tree = {selected_attribute: {}}
  attributes.remove(selected_attribute)
  
  for value in df[selected_attribute].unique():
    subset = df[df[selected_attribute] == value]
    if len(subset) == 0:
      tree[selected_attribute][value] = df[df.keys()[-1]].value_counts().idxmax()
    else:
      maxDepth-=1
      if maxDepth == 0:
        tree[selected_attribute][value] = df[selected_attribute].mode()[0]
      else:
        tree[selected_attribute][value] = DecisionTreeClassifier(subset, attributes.copy(), maxDepth)
  
  return tree

def predict(instance, tree):
  for node, value in tree.items():
    if instance[node] not in value:
      return "NotALeaf"
    value = value[instance[node]]
    if isinstance(value, dict):
      return predict(instance, value)
    else:
      return value
    
def evaluate(df, trees):
    predicted_labels = []
    correct_prediction = 0
    for i in range(len(df)):
        instance = df.iloc[i, :-1] 
        predictions = [predict(instance, tree) for tree in trees]
        final_prediction = max(set(predictions), key=predictions.count)
        predicted_labels.append(final_prediction)
        if df.iloc[i, -1] == final_prediction:
            correct_prediction += 1
    accuracy = correct_prediction / len(df)
    return accuracy


def bagging_decision_trees(train_df, test_df, num_trees, sample_size):
  each_tree_train_errors = []
  each_tree_test_errors = []
  trees = []
  single_tree_test_error = 0.0
  for _ in range(num_trees):
    train_sample = train_df.sample(n=sample_size, replace = False)
    test_sample = test_df.sample(n=sample_size, replace = False)
    tree = DecisionTreeClassifier(train_sample)
    trees.append(tree)
    
    train_accuracy = evaluate(train_sample, trees)
    test_accuracy = evaluate(test_sample, trees)
    each_tree_train_errors.append(1 - train_accuracy)
    each_tree_test_errors.append(1 - test_accuracy)
    if _ == 1:
      single_tree_test_error += (1 - test_accuracy)

  return each_tree_train_errors, each_tree_test_errors, single_tree_test_error

#replace numeric attribute values with median threshold
def median_thresholding(df, attribute):
  threshold = df[attribute].median()
  df[attribute] = (df[attribute] >= threshold).astype(int)

if __name__ == "__main__":
    
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
    
    train_errors = []
    iterations = 100
    numberOfTrees = 500
    test_errors=[]
    single_tree_test_errors = []
    for iter in range(iterations):
        bag_train_error, bag_test_error, single_tree_test_error = bagging_decision_trees(df_train, df_test, numberOfTrees, 1000)
        train_errors.append(bag_train_error)
        test_errors.append(bag_test_error)
        single_tree_test_errors.append(single_tree_test_error)

    value = np.array(df_test.iloc[:, -1].tolist())
    value[value == 'yes'] = 1
    value[value == 'no'] = -1
    print(value)
    print(single_tree_test_errors)
    value = value.astype(int)
    bias = np.mean(np.square(single_tree_test_errors - value))
    mean = np.mean(single_tree_test_errors) 
    variance = np.sum(np.square(single_tree_test_errors - mean)) / (len(df_test) - 1)
    print("bias =",bias)
    print("variance =",variance)
    print('bias + variance in 100 single tree predictor = ', bias+variance)
    for row in range(len(test_errors)):
      test_errors[row] = np.mean(test_errors[row])
    test_errors = np.sum(test_errors,axis=0) / (iterations * numberOfTrees)
    bias = np.mean(np.square(test_errors - value))
    mean = np.mean(test_errors)
    variance = np.sum(np.square(test_errors - mean)) / (len(df_test) - 1)
    print("bias =",bias)
    print("variance =",variance)
    print('bias + variance in 100 bagged tree predictor =', bias+variance)