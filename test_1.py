from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import pipeline

import os

# Set your desired cache directory
os.environ['HF_HOME'] = 'D:\Repositories\multilingual_mice\.cache'

# Local imports
from mmice.stage_one import run_train_editor 
from mmice.utils import get_args, get_dataset_reader, get_device


if __name__ == '__main__':

    args = get_args("stage1")
    predictor = pipeline("text-classification",
                         model=f"trained_predictors/{args.meta.task}/model",
                         device=get_device(),
                         max_length=512,
                         padding=True,
                         truncation=True,
                         top_k=None)
    dataset_reader = get_dataset_reader(args.meta.task)
    with logging_redirect_tqdm():
        run_train_editor(predictor, dataset_reader, args)