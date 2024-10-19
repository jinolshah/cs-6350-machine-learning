import getData
from bag import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
  
  train_data, test_data = getData.getData()

  numeric_attributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
  for numeric_attr in numeric_attributes:
    median_thr(train_data, numeric_attr)
    median_thr(test_data, numeric_attr)
  
  train_errors = []
  test_errors=[]
  iterations = 500
  for i in range(iterations):
    bag_train_error, bag_test_error = bagging_decision_trees(train_data, test_data, i, 1000)
    train_errors.append(bag_train_error)
    test_errors.append(bag_test_error)
  print(train_errors)
  print(test_errors)
  plt.figure(figsize=(10, 6))
  plt.plot(iterations, train_errors, label='Train Errors', color='blue')
  plt.plot(iterations, test_errors, label='Test Errors', color='red')
  plt.title('Train and Test Errors vs. Iteration')
  plt.xlabel('Iteration')
  plt.ylabel('Train and Test Errors')
  plt.legend()
  plt.grid(True)
  plt.show()
