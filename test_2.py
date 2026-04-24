from tqdm.contrib.logging import logging_redirect_tqdm
from pathlib import Path
import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


# Set your desired cache directory
os.environ["HF_HOME"] = "/content/drive/MyDrive/mmice_data"

# Local imports
from mmice.stage_two import run_edit_test
from mmice.utils import get_args
from mmice.stage_two import run_edit_test
from mmice.utils import get_args, get_device
from mmice.predictor_wrapper import MultimodalPredictorWrapper
from mmice.biomed_clip import BiomedCLIPClassifier

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
    args = get_args("stage2")
    device = get_device()

    model = BiomedCLIPClassifier(num_classes=8, device=device)
    checkpoint_path = hf_hub_download(
        repo_id="BoSsa-Projects/BiomedCLIP-MIMIC-CXR-Predictor",
        filename="best_model.pth",
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        model_max_length=256,
    )

    predictor = MultimodalPredictorWrapper(
        model=model,
        tokenizer=tokenizer,
        label2id=LABEL2ID,
        device=device,
        image_key="pixel_values",
        problem_type="multi_label_classification",
    )

    with logging_redirect_tqdm():
        run_edit_test(args, predictor=predictor)


if __name__ == "__main__":
    main()
