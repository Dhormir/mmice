import pandas as pd
from datasets import Dataset

# Task loader script in case someone wants to expand and add a new task
# just create a loader here that returns a Hugginface Dataset object
# with text and label columns

def load_chilean_hate(data_files=None, column_names=['text', 'Odio']):
    kwargs = {"encoding" : "utf-8", "index_col" : 0, "sep" : ","}
    df_list = [pd.read_csv(data_file, **kwargs) for data_file in data_files]
    df = pd.concat(df_list)[column_names]
    
    data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    data = data.rename_column('Odio', 'label')
    data = data.shuffle(42)
    return data.train_test_split(train_size=.89)