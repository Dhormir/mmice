from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import pipeline
from pathlib import Path
import os

# Set cache to current repo
os.environ["HF_HOME"] = str(Path.cwd() / ".cache")

# Local imports
from mmice.stage_one import run_train_editor
from mmice.utils import get_args, get_dataset_reader, get_device


def main():
    args = get_args("stage1")
    predictor = pipeline(
        "text-classification",
        model=f"{args.meta.predictors_dir}/{args.meta.task}/model",
        device=get_device(),
        max_length=512,
        padding=True,
        truncation=True,
        top_k=None,
    )
    dataset_reader = get_dataset_reader(args.meta.task, data_dir=args.meta.data_dir)
    with logging_redirect_tqdm():
        run_train_editor(predictor, dataset_reader, args)


if __name__ == "__main__":
    main()
