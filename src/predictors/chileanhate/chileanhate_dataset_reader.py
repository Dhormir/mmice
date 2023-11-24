from typing import Dict, List, Optional
import logging

from allennlp.data import Tokenizer
from overrides import overrides
from nltk.tree import Tree

from sklearn.model_selection import train_test_split

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from allennlp.common.checks import ConfigurationError

from pathlib import Path
from itertools import chain
import os.path as osp
import tarfile
import pandas as pd
import numpy as np
import math

from src.predictors.predictor_utils import clean_text 
logger = logging.getLogger(__name__)

TRAIN_VAL_SPLIT_RATIO = 0.85
        
def get_label(p):
    assert 1 == p or 0 == p
    return str(p)

@DatasetReader.register("chileanhate")
class ChileanHateDatasetReader(DatasetReader):

    #TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz' 
    TRAIN_DIR = 'src/predictors/chileanhate/data/'
    TEST_DIR = 'src/predictors/chileanhate/data/'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None, # type: ignore
                 tokenizer: Optional[Tokenizer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or \
                {"tokens": SingleIdTokenIndexer()}

        self.random_seed = 42 # numpy random seed
    
    def get_inputs(self, file_path, return_labels=False):
        np.random.seed(self.random_seed)
        df_train = pd.read_csv(self.TRAIN_DIR + '/tweets_train.csv',
                               encoding='utf-8',
                               index_col=0,
                               sep=',')
        df_test = pd.read_csv(self.TEST_DIR + '/tweets_test.csv',
                              encoding='utf-8',
                              index_col=0,
                              sep=',')

        df = pd.concat([df_train, df_test])
        df = df.sample(frac=1, random_state=self.random_seed)

        strings = []
        labels = []

        if 'train' in file_path:
            print(f'len pre split train: {len(df)}')
            _, df =  train_test_split(df,
                                      test_size=TRAIN_VAL_SPLIT_RATIO,
                                      random_state=self.random_seed)
            print(f'len post split train: {len(df)}')
        else:
            print(f'len pre split test: {len(df)}')
            df, _ =  train_test_split(df,
                                      test_size=TRAIN_VAL_SPLIT_RATIO,
                                      random_state=self.random_seed)
            print(f'len post split test: {len(df)}')

        for idx, (index, row) in enumerate(df.iterrows()):
            labels.append(get_label(row['Odio']))
            strings.append(clean_text(row['text'], special_chars=["<br />", "\t"]))

        if return_labels:
            return strings, labels
        return strings 

    @overrides
    def _read(self, file_path):
        np.random.seed(self.random_seed)
        df_train = pd.read_csv(self.TRAIN_DIR + '/tweets_train.csv',
                               encoding='utf-8',
                               index_col=0,
                               sep=',')
        df_test = pd.read_csv(self.TEST_DIR + '/tweets_test.csv',
                              encoding='utf-8',
                              index_col=0,
                              sep=',')

        df = pd.concat([df_train, df_test])
        df = df.sample(frac=1, random_state=self.random_seed)

        if 'train' in file_path:
            df, _ =  train_test_split(df,
                                      test_size=TRAIN_VAL_SPLIT_RATIO,
                                      random_state=self.random_seed)
        else:
            _, df =  train_test_split(df,
                                      test_size=TRAIN_VAL_SPLIT_RATIO,
                                      random_state=self.random_seed)

        for index, row in df.iterrows():
            label = get_label(row['Odio'])
            yield self.text_to_instance(
                    clean_text(row['text'], special_chars=["<br />", "\t"]),
                    label)

    def text_to_instance(
            self, string: str, label:str = None) -> Optional[Instance]:
        tokens = self._tokenizer.tokenize(string)
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)