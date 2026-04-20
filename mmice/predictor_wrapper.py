"""
Wrapper to make a custom multimodal predictor (e.g. BiomedCLIP)
compatible with MMiCE's GradientMasker and stage_one pipeline.

MMiCE expects a predictor with:
    - predictor.model            → nn.Module with forward(**kwargs) → {"logits": tensor}
    - predictor.model.config     → has label2id, id2label, problem_type
    - predictor.tokenizer        → HF-style tokenizer with __call__, all_special_tokens, etc.
    - predictor.device           → torch.device
    - predictor(text)            → list of [{"label": str, "score": float}, ...]
"""

import torch
import torch.nn as nn
from munch import Munch
from torchvision import transforms

BIOMED_CLIP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),  # L → RGB
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


class MultimodalPredictorWrapper:
    """
    Wraps a custom multimodal classifier to look like a HuggingFace pipeline.

    Args:
        model: nn.Module that takes (input_ids, pixel_values/images) → logits
        tokenizer: text tokenizer (HF-style with __call__, all_special_tokens, etc.)
        label2id: dict mapping label names to indices, e.g. {"Pneumonia": 0, ...}
        device: torch device
        image_key: kwarg name the underlying model expects for images.
                   The wrapper maps "images" → this key.
        problem_type: "single_label_classification" or "multi_label_classification"
        embedding_attr: dotted path to the text embedding layer on the model,
                        for GradientMasker hooks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        label2id,
        device="cpu",
        image_key="pixel_values",
        problem_type="multi_label_classification",
        embedding_attr="text_encoder",
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.embedding_attr = embedding_attr

        # Wrap the model so its forward returns {"logits": ...}
        # and maps "images" kwarg to whatever the model actually expects
        self.model = _WrappedModel(
            model=model,
            label2id=label2id,
            problem_type=problem_type,
            image_key=image_key,
        )

    def __call__(self, text, images=None, **kwargs):
        """
        Mimics pipeline(text) → list of predictions.
        For multimodal, this is text-only inference (no image).
        Returns list of [{"label": str, "score": float}, ...] per sample.
        """
        if isinstance(text, str):
            text = [text]

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", 512),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            forward_kwargs = dict(tokenized)
            if images is not None:
                if hasattr(images, "to"):
                    images = images.to(self.device)
                # Add batch dim if single image
                if images.dim() == 3:
                    images = images.unsqueeze(0)
                forward_kwargs["images"] = images
            outputs = self.model(**forward_kwargs)
            logits = outputs["logits"]

        id2label = self.model.config.id2label
        results = []
        for sample_logits in logits:
            if self.model.config.problem_type == "multi_label_classification":
                probs = torch.sigmoid(sample_logits)
            else:
                probs = torch.softmax(sample_logits, dim=-1)

            preds = []
            for idx in range(len(probs)):
                preds.append(
                    {
                        "label": id2label[idx],
                        "score": probs[idx].item(),
                    }
                )
            # Sort by score descending
            preds.sort(key=lambda x: x["score"], reverse=True)
            results.append(preds)

        if isinstance(text, str):
            return results[0]

        return results


class _WrappedModel(nn.Module):
    """
    Thin wrapper that:
    1. Adds a .config with label2id / id2label / problem_type
    2. Maps "images" kwarg to the model's actual image kwarg name
    3. Ensures forward returns {"logits": tensor}
    """

    def __init__(self, model, label2id, problem_type, image_key="pixel_values"):
        super().__init__()
        self.model = model
        self.image_key = image_key

        # Create a config-like object for MMiCE compatibility
        id2label = {v: k for k, v in label2id.items()}
        self.config = Munch(
            label2id=label2id,
            id2label=id2label,
            problem_type=problem_type,
        )

    def forward(self, input_ids=None, attention_mask=None, images=None, **kwargs):
        # Build kwargs for the underlying model
        forward_kwargs = {}
        if input_ids is not None:
            forward_kwargs["input_ids"] = input_ids
        if images is not None:
            if images.dim() == 3:
                images = images.unsqueeze(0)
            forward_kwargs[self.image_key] = images

        out = self.model(**forward_kwargs)

        # Normalize output to {"logits": tensor}
        if isinstance(out, dict):
            return out
        elif isinstance(out, torch.Tensor):
            return {"logits": out}
        else:
            return {"logits": out[0]}

    # Expose the underlying model's named_parameters, etc.
    # so GradientMasker can iterate and set requires_grad
    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        return self.model.zero_grad(*args, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self
