import torch
import torch.nn as nn
import open_clip


class BiomedCLIPClassifier(nn.Module):
    def __init__(self, num_classes=8, device="cpu"):
        super().__init__()
        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        print(f"Loading base BiomedCLIP from {model_name}...")

        # Load base model structure
        clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name, device=device
        )
        self.visual = clip_model.visual
        self.text_encoder = clip_model.text

        # Dynamic dimension calculation to match saved weights
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224).to(device)
            vis_dim = self.visual(dummy_img).shape[1]
            dummy_text = torch.randint(0, 1000, (1, 256)).to(device)
            text_dim = self.text_encoder(dummy_text).shape[1]

        self.classifier = nn.Linear(vis_dim + text_dim, num_classes)

    def forward(self, pixel_values, input_ids):
        # Normalize features as per Contrastive Training standards
        text_features = self.text_encoder(input_ids)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        if pixel_values is not None:
            vis_features = self.visual(pixel_values)
            vis_features = vis_features / vis_features.norm(dim=-1, keepdim=True)
        else:
            vis_dim = self.classifier.in_features - text_features.shape[1]
            vis_features = torch.zeros(
                text_features.shape[0], vis_dim, device=input_ids.device
            )

        combined_features = torch.cat((vis_features, text_features), dim=1)
        return self.classifier(combined_features)
