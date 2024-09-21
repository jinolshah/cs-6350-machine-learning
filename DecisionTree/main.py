import os
from tree import *
from hyperparameter import entropy_weighted, majority_weighted, gini_weighted, avg_err, extract_examples 

def print_results(max_depth, database_name, train_data, test_data, categoricals):
    train_storage = []
    test_storage = []
    for depth in range(1, max_depth+1):
        e_tree = DecisionTree(depth, 0, train_data, entropy_weighted, None, categoricals)
        m_tree = DecisionTree(depth, 0, train_data, majority_weighted, None, categoricals)
        g_tree = DecisionTree(depth, 0, train_data, gini_weighted, None, categoricals)

        train_storage.append((depth, avg_err(e_tree, train_data), avg_err(m_tree, train_data), avg_err(g_tree, train_data)))

        test_storage.append((depth, avg_err(e_tree, test_data), avg_err(m_tree, test_data), avg_err(g_tree, test_data)))

    print(f"{database_name}:", '\n')

    print(f"{'Depth':<5} {'Entropy Train Error':>20} {'Majority Train Error':>20} {'Gini Train Error':>20}")
    print("-" * 65)
    for a, b, c, d in train_storage:
        print(f"{a:<5} {b:>20.3f} {c:>20.3f} {d:>20.3f}")

    print()

    print(f"{'Depth':<5} {'Entropy Test Error':>20} {'Majority Test Error':>20} {'Gini Test Error':>20}")
    print("-" * 65)
    for a, b, c, d in test_storage:
        print(f"{a:<5} {b:>20.3f} {c:>20.3f} {d:>20.3f}")

    print('\n')



if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    car_train_data_file = os.path.join(script_dir, "car/train.csv")
    car_test_data_file = os.path.join(script_dir, "car/test.csv")
    bank_train_data_file = os.path.join(script_dir, "bank/train.csv")
    bank_test_data_file = os.path.join(script_dir, "bank/train.csv")

    car_test_samples, _, categoricals = extract_examples(car_test_data_file, True)
    car_train_samples, _, categoricals = extract_examples(car_train_data_file, True)
    print_results(6, 'Car data', car_train_samples, car_test_samples, categoricals)

    bank_test_samples_unknown, _, categoricals = extract_examples(bank_test_data_file, True)
    bank_train_samples_unknown, _, categoricals = extract_examples(bank_train_data_file, True)
    print_results(16, 'Bank data (unknown is a value)', bank_train_samples_unknown, bank_test_samples_unknown, categoricals)

    bank_train_samples_unknown_replaced, _, categoricals = extract_examples(bank_train_data_file, False)
    print_results(16, 'Bank data (unknown replaced with majority)', bank_train_samples_unknown_replaced, bank_test_samples_unknown, categoricals)
