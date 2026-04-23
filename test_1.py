from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import pipeline
from pathlib import Path
import os

# Set cache to current repo
os.environ["HF_HOME"] = "/content/drive/MyDrive/mmice_data"

# Local imports
from mmice.stage_one import run_train_editor
from mmice.predictor_wrapper import MultimodalPredictorWrapper
from mmice.utils import get_args, get_dataset_reader, get_device

from mmice.biomed_clip import BiomedCLIPClassifier
from huggingface_hub import hf_hub_download
import torch
from transformers import AutoTokenizer

LABEL2ID = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Edema": 3,
    "Lung Opacity": 4,
    "Pleural Effusion": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
}


def main():
    args = get_args("stage1")
    device = get_device()

    # predictor = pipeline(
    #     "text-classification",
    #     model=f"{args.meta.predictors_dir}/{args.meta.task}/model",
    #     device=get_device(),
    #     max_length=512,
    #     padding=True,
    #     truncation=True,
    #     top_k=None,
    # )

    # Load your multimodal predictor
    model = BiomedCLIPClassifier(num_classes=8, device=device)
    checkpoint_path = hf_hub_download(
        repo_id="BoSsa-Projects/BiomedCLIP-MIMIC-CXR-Predictor",
        filename="best_model.pth",
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    # BiomedCLIP uses PubMedBERT under the hood
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        model_max_length=256,
    )

    # Wrap it to look like a HuggingFace pipeline
    predictor = MultimodalPredictorWrapper(
        model=model,
        tokenizer=tokenizer,
        label2id=LABEL2ID,
        device=device,
        image_key="pixel_values",  # what BiomedCLIPClassifier.forward expects
        problem_type="multi_label_classification",
    )
    dataset_reader = get_dataset_reader(args.meta.task, data_dir=args.meta.data_dir)
    with logging_redirect_tqdm():
        run_train_editor(predictor, dataset_reader, args)


if __name__ == "__main__":
    main()
