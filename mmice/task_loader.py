import pandas as pd
from datasets import Dataset, Sequence, Value, Features

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

def load_42k_hcuch(data_files=None,
                   column_names=['hallazgos', 'impresion',
                                 'condensacion', 'nodulos', 'quistes']):
    kwargs = {"encoding" : "utf-8",
              "sep" : "|",
              "dtype":
                  {'condensacion': int, 'nodulos': int, 'quistes': int}}

    def combine_columns(example):
        example["label"] = [example["condensacion"], example["nodulos"], example["quistes"]]
        return example

    df_list = [pd.read_csv(data_file, **kwargs) for data_file in data_files]
    df = pd.concat(df_list)[column_names]
    df['text'] = df['hallazgos'] # + " " +  df['impresion']
    
    data = Dataset.from_pandas(df).remove_columns(['hallazgos', 'impresion', '__index_level_0__'])
    data = data.map(combine_columns)
    new_features = data.features.copy()
    new_features["label"] = Sequence(feature=Value("int8"), length=3)

    data = data.cast(new_features)
    data = data.remove_columns(['condensacion', 'nodulos', 'quistes'])
    data = data.shuffle(42)
    # For multilabel models we will make it focus only on the highest probability label
    return data.train_test_split(train_size=.75)

def load_semeval_hate(data_files=None, column_names=['text', 'HS']):
    kwargs = {"encoding" : "utf-8"}
    df_list = [pd.read_csv(data_file, **kwargs) for data_file in data_files]
    df = pd.concat(df_list)[column_names]
    
    data = Dataset.from_pandas(df).remove_columns(["__index_level_0__"])
    data = data.rename_column('HS', 'label')
    data = data.shuffle(42)
    return data.train_test_split(train_size=.75)