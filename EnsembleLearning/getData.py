import os
import pandas as pd
import dataset

def getData():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bank_train_data_file = os.path.join(script_dir, "bank/train.csv")
    bank_test_data_file = os.path.join(script_dir, "bank/train.csv")

    train_data = pd.read_csv(bank_train_data_file)
    train_data.columns = dataset.bank_columns
    test_data = pd.read_csv(bank_test_data_file)
    test_data.columns = dataset.bank_columns

    return train_data, test_data