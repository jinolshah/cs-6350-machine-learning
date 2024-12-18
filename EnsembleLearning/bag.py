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

def median_thr(df, attribute):
  threshold = df[attribute].median()
  df[attribute] = (df[attribute] >= threshold).astype(int)