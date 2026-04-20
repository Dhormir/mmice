"""
Tests for gradient-based text masking with a multimodal predictor.

Scenario: A multimodal classifier (text + images → class logits) is the predictor.
GradientMasker hooks into its text embeddings, computes saliency, and masks
only text tokens. The masked text + images are then fed to the multimodal T5 editor.

Run:
    uv run pytest mmice/tests/test_multimodal_masker.py -v
"""

import math
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch import backends
from transformers import T5ForConditionalGeneration, T5Config

from mmice.multimodal import (
    VisionEncoder,
    PerceiverResampler,
    MultimodalT5ForConditionalGeneration,
)


# ═══════════════════════════════════════════════════════════
# Fake multimodal predictor (classifier)
# ═══════════════════════════════════════════════════════════


class FakeMultimodalPredictor(nn.Module):
    """
    Minimal multimodal classifier: text + image → class logits.
    Mirrors a real predictor pipeline but small enough to test with.

    Architecture:
        text  → embedding → mean-pool → (text_dim,)
        image → vision_encoder → mean-pool → project → (text_dim,)
        concat [text_repr, visual_repr] → classifier → (num_classes,)
    """

    def __init__(
        self,
        vocab_size=128,
        text_embed_dim=64,
        img_size=32,
        patch_size=8,
        vision_embed_dim=32,
        num_classes=3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Text pathway — this is what GradientMasker hooks into
        self.embeddings = nn.Embedding(vocab_size, text_embed_dim)

        # Vision pathway
        self.vision_encoder = VisionEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=vision_embed_dim,
            depth=1,
            num_heads=4,
        )
        self.visual_projection = nn.Linear(vision_embed_dim, text_embed_dim)

        # Classifier head (text_dim + text_dim → num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(text_embed_dim * 2, text_embed_dim),
            nn.ReLU(),
            nn.Linear(text_embed_dim, num_classes),
        )

    def forward(self, input_ids, images=None, attention_mask=None):
        # Text encoding
        text_embeds = self.embeddings(input_ids)  # (B, seq_len, dim)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            text_repr = (text_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            text_repr = text_embeds.mean(dim=1)  # (B, dim)

        # Visual encoding
        if images is not None:
            visual_features = self.vision_encoder(images)  # (B, patches, vis_dim)
            visual_features = self.visual_projection(
                visual_features
            )  # (B, patches, dim)
            visual_repr = visual_features.mean(dim=1)  # (B, dim)
        else:
            visual_repr = torch.zeros_like(text_repr)

        # Fuse and classify
        fused = torch.cat([text_repr, visual_repr], dim=-1)  # (B, dim*2)
        logits = self.classifier(fused)  # (B, num_classes)
        return {"logits": logits}


# ═══════════════════════════════════════════════════════════
# GradientMasker-compatible functions
# (replicate the exact pattern from gradient_masker.py)
# ═══════════════════════════════════════════════════════════


def register_embedding_hooks(predictor_model):
    """
    Replicates GradientMasker._register_embedding_gradient_hooks.
    Hooks into predictor.model.base_model.embeddings (here: predictor.embeddings).
    """
    embedding_gradients = []

    def hook_layers(module, grad_in, grad_out):
        grads = grad_out[0]
        embedding_gradients.append(grads)

    handle = predictor_model.embeddings.register_full_backward_hook(hook_layers)
    return embedding_gradients, [handle]


def get_gradients_by_prob(predictor, input_ids, pred_idx, images=None):
    """
    Replicates GradientMasker._get_gradients_by_prob for a multimodal predictor.
    Computes gradient of logits[pred_idx] w.r.t. text embeddings.

    Returns grad_dict in same format: {"grad_input_1": np.ndarray}
    """
    # Save and override requires_grad (like GradientMasker lines 149-154)
    original_requires_grad = {}
    for name, param in predictor.named_parameters():
        original_requires_grad[name] = param.requires_grad
        param.requires_grad = True

    embedding_gradients, hooks = register_embedding_hooks(predictor)

    with backends.cudnn.flags(enabled=True):
        outputs = predictor(input_ids=input_ids, images=images)
        prob = outputs["logits"][0][pred_idx]
        predictor.zero_grad()
        prob.backward()

    for hook in hooks:
        hook.remove()

    # Build grad_dict (like GradientMasker lines 177-180)
    grad_dict = {}
    for idx, grad in enumerate(embedding_gradients):
        key = "grad_input_" + str(idx + 1)
        grad_dict[key] = grad.detach().cpu().numpy()

    # Restore requires_grad (like GradientMasker lines 183-184)
    for name, param in predictor.named_parameters():
        param.requires_grad = original_requires_grad[name]

    return grad_dict


def get_gradient_magnitudes(grad_dict, grad_type="normal_l1", sign_direction=1):
    """
    Replicates GradientMasker._get_gradient_magnitudes (lines 375-432).
    """
    grad = grad_dict["grad_input_1"][0]  # (seq_len, embed_dim)

    if grad_type == "normal_l1":
        grad_signed = np.sum(abs(grad), axis=1)
        grad_magnitudes = grad_signed.copy()
    elif grad_type == "normal_signed":
        grad_signed = np.sum(grad, axis=1)
        grad_magnitudes = sign_direction * grad_signed
    elif grad_type == "normal_l2":
        grad_signed = np.array([g.dot(g) for g in grad])
        grad_magnitudes = grad_signed.copy()
    else:
        raise ValueError(f"Unsupported grad_type: {grad_type}")

    return grad_signed, grad_magnitudes


def get_mask_indices(grad_magnitudes, mask_frac=0.5, special_tok_indices=None):
    """
    Replicates the ranking logic from GradientMasker.get_important_editor_tokens
    (lines 543-575): argsort descending, skip specials, return top-k.
    """
    special_tok_indices = special_tok_indices or set()
    ordered = np.argsort(grad_magnitudes)[::-1]
    filtered = [i for i in ordered if i not in special_tok_indices]
    k = math.ceil(mask_frac * len(filtered))
    return filtered[:k]


def apply_sentinel_mask(input_ids, mask_indices, sentinel_id=1):
    """Apply mask by replacing selected positions with a sentinel token."""
    masked = input_ids.clone()
    for idx in mask_indices:
        masked[0, idx] = sentinel_id
    return masked


# ═══════════════════════════════════════════════════════════
# Tiny multimodal T5 editor factory (same as test_multimodal.py)
# ═══════════════════════════════════════════════════════════

TINY_T5_CONFIG = T5Config(
    vocab_size=128,
    d_model=64,
    d_ff=128,
    d_kv=16,
    num_heads=4,
    num_layers=6,
    num_decoder_layers=6,
    decoder_start_token_id=0,
    eos_token_id=1,
    pad_token_id=0,
)
TINY_VIS = {
    "img_size": 32,
    "patch_size": 8,
    "embed_dim": 32,
    "depth": 1,
    "num_heads": 4,
}


def _make_tiny_editor(**overrides):
    kwargs = dict(
        t5_model_name="__tiny__",
        vision_config=TINY_VIS,
        use_perceiver=True,
        num_perceiver_latents=4,
        perceiver_depth=1,
    )
    kwargs.update(overrides)
    _orig = MultimodalT5ForConditionalGeneration.__init__

    def _patched(self, **kw):
        nn.Module.__init__(self)
        self.config = TINY_T5_CONFIG
        self.t5 = T5ForConditionalGeneration(TINY_T5_CONFIG)
        vc = kw.get("vision_config", {})
        ved = vc.get("embed_dim", 768)
        self.vision_encoder = VisionEncoder(
            img_size=vc.get("img_size", 224),
            patch_size=vc.get("patch_size", 16),
            embed_dim=ved,
            depth=vc.get("depth", 12),
            num_heads=vc.get("num_heads", 12),
        )
        self.use_perceiver = kw.get("use_perceiver", True)
        if self.use_perceiver:
            self.perceiver = PerceiverResampler(
                dim=ved,
                depth=kw.get("perceiver_depth", 6),
                num_latents=kw.get("num_perceiver_latents", 64),
                num_heads=4,
            )
        hs = self.config.d_model
        self.visual_projection = nn.Linear(ved, hs) if ved != hs else nn.Identity()
        dcal = kw.get("decoder_cross_attn_layers") or list(
            range(0, self.config.num_decoder_layers, 3)
        )
        self.decoder_cross_attn_layers = dcal
        self._wrap_decoder_layers()
        if kw.get("freeze_vision", False):
            self._freeze_vision_encoder()
        if kw.get("freeze_t5", False):
            self._freeze_t5()

    MultimodalT5ForConditionalGeneration.__init__ = lambda self, **kw: _patched(
        self, **kw
    )
    try:
        m = MultimodalT5ForConditionalGeneration(**kwargs)
    finally:
        MultimodalT5ForConditionalGeneration.__init__ = _orig
    return m


def fake_ids(batch=1, seq_len=10):
    return torch.randint(2, 128, (batch, seq_len))


# ═══════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════


@pytest.fixture
def predictor():
    """Multimodal predictor (classifier)."""
    return FakeMultimodalPredictor()


@pytest.fixture
def editor():
    """Multimodal T5 editor."""
    return _make_tiny_editor()


# ═══════════════════════════════════════════════════════════
# Tests: Gradient capture on the multimodal predictor
# ═══════════════════════════════════════════════════════════


class TestPredictorGradientCapture:
    """Hook into the multimodal predictor's text embeddings and capture gradients."""

    def test_grad_dict_has_expected_key(self, predictor):
        grad_dict = get_gradients_by_prob(
            predictor,
            fake_ids(),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        assert "grad_input_1" in grad_dict

    def test_grad_shape_matches_text_seq_len(self, predictor):
        """Gradients should be (1, text_seq_len, embed_dim) — no visual tokens."""
        seq_len = 12
        grad_dict = get_gradients_by_prob(
            predictor,
            fake_ids(seq_len=seq_len),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        grad = grad_dict["grad_input_1"]
        assert grad.shape == (
            1,
            seq_len,
            64,
        ), f"Expected (1, {seq_len}, 64), got {grad.shape}"

    def test_grad_shape_excludes_visual_patches(self, predictor):
        """
        The image has (32/8)^2 = 16 patches, but the gradient should only
        cover the 10 text tokens — visual patches must NOT appear in grad_input_1.
        """
        text_len = 10
        grad_dict = get_gradients_by_prob(
            predictor,
            fake_ids(seq_len=text_len),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        assert grad_dict["grad_input_1"].shape[1] == text_len

    def test_grads_nonzero(self, predictor):
        grad_dict = get_gradients_by_prob(
            predictor,
            fake_ids(),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        assert np.abs(grad_dict["grad_input_1"]).sum() > 0

    def test_grads_no_nans(self, predictor):
        grad_dict = get_gradients_by_prob(
            predictor,
            fake_ids(),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        assert not np.isnan(grad_dict["grad_input_1"]).any()

    def test_hooks_cleaned_up(self, predictor):
        n_before = len(predictor.embeddings._backward_hooks)
        _ = get_gradients_by_prob(
            predictor,
            fake_ids(),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        assert len(predictor.embeddings._backward_hooks) == n_before

    def test_requires_grad_restored(self, predictor):
        """Predictor parameters should be restored after gradient capture."""
        orig = {n: p.requires_grad for n, p in predictor.named_parameters()}
        _ = get_gradients_by_prob(
            predictor,
            fake_ids(),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        for name, param in predictor.named_parameters():
            assert (
                param.requires_grad == orig[name]
            ), f"requires_grad changed for {name}"

    def test_weights_unchanged(self, predictor):
        """Gradient capture must not update weights."""
        before = {n: p.clone() for n, p in predictor.named_parameters()}
        _ = get_gradients_by_prob(
            predictor,
            fake_ids(),
            pred_idx=0,
            images=torch.randn(1, 3, 32, 32),
        )
        for name, param in predictor.named_parameters():
            assert torch.equal(param.data, before[name]), f"{name} changed"


# ═══════════════════════════════════════════════════════════
# Tests: Images influence text saliency in the predictor
# ═══════════════════════════════════════════════════════════


class TestImageInfluenceOnTextSaliency:
    """The whole point of multimodal masking: images should change which text gets masked."""

    def test_different_images_different_magnitudes(self, predictor):
        ids = fake_ids()
        grad_a = get_gradients_by_prob(
            predictor, ids.clone(), 0, torch.randn(1, 3, 32, 32)
        )
        grad_b = get_gradients_by_prob(
            predictor, ids.clone(), 0, torch.randn(1, 3, 32, 32)
        )
        _, mag_a = get_gradient_magnitudes(grad_a, "normal_l1")
        _, mag_b = get_gradient_magnitudes(grad_b, "normal_l1")
        assert not np.allclose(
            mag_a, mag_b, atol=1e-6
        ), "Different images produced identical text saliency"

    def test_with_vs_without_images_differs(self, predictor):
        ids = fake_ids()
        grad_with = get_gradients_by_prob(
            predictor, ids.clone(), 0, torch.randn(1, 3, 32, 32)
        )
        grad_without = get_gradients_by_prob(predictor, ids.clone(), 0, images=None)
        _, mag_with = get_gradient_magnitudes(grad_with, "normal_l1")
        _, mag_without = get_gradient_magnitudes(grad_without, "normal_l1")
        assert not np.allclose(
            mag_with, mag_without, atol=1e-6
        ), "Images had zero effect on text saliency"

    def test_different_images_shift_magnitude_distribution(self, predictor):
        """Different images should shift the saliency magnitude distribution,
        even if the top-k ranking doesn't always change."""
        ids = fake_ids(seq_len=15)
        magnitudes_list = []
        for _ in range(5):
            gd = get_gradients_by_prob(
                predictor, ids.clone(), 0, torch.randn(1, 3, 32, 32)
            )
            _, mag = get_gradient_magnitudes(gd, "normal_l1")
            magnitudes_list.append(mag)

        # Check that at least one pair of magnitude vectors differs
        all_same = all(
            np.allclose(magnitudes_list[0], m, atol=1e-6) for m in magnitudes_list[1:]
        )
        assert (
            not all_same
        ), "5 different images all produced identical magnitude vectors"

    def test_different_classes_different_saliency(self, predictor):
        """Gradient of class 0 vs class 2 should produce different saliency maps."""
        ids = fake_ids()
        images = torch.randn(1, 3, 32, 32)
        grad_c0 = get_gradients_by_prob(
            predictor, ids.clone(), pred_idx=0, images=images.clone()
        )
        grad_c2 = get_gradients_by_prob(
            predictor, ids.clone(), pred_idx=2, images=images.clone()
        )
        _, mag_c0 = get_gradient_magnitudes(grad_c0, "normal_l1")
        _, mag_c2 = get_gradient_magnitudes(grad_c2, "normal_l1")
        assert not np.allclose(
            mag_c0, mag_c2, atol=1e-6
        ), "Different target classes produced identical saliency"


# ═══════════════════════════════════════════════════════════
# Tests: Saliency aggregation (L1 / L2 / signed)
# ═══════════════════════════════════════════════════════════


class TestSaliencyAggregation:

    @pytest.mark.parametrize("grad_type", ["normal_l1", "normal_l2", "normal_signed"])
    def test_magnitude_shape(self, predictor, grad_type):
        seq_len = 10
        gd = get_gradients_by_prob(
            predictor, fake_ids(seq_len=seq_len), 0, torch.randn(1, 3, 32, 32)
        )
        sd = 1 if "signed" in grad_type else None
        _, mag = get_gradient_magnitudes(gd, grad_type, sign_direction=sd or 1)
        assert mag.shape == (seq_len,)

    @pytest.mark.parametrize("grad_type", ["normal_l1", "normal_l2"])
    def test_non_negative(self, predictor, grad_type):
        gd = get_gradients_by_prob(predictor, fake_ids(), 0, torch.randn(1, 3, 32, 32))
        _, mag = get_gradient_magnitudes(gd, grad_type)
        assert (mag >= 0).all()

    def test_not_degenerate(self, predictor):
        gd = get_gradients_by_prob(
            predictor, fake_ids(seq_len=15), 0, torch.randn(1, 3, 32, 32)
        )
        _, mag = get_gradient_magnitudes(gd, "normal_l1")
        assert mag.std() > 0, "All tokens have identical saliency"


# ═══════════════════════════════════════════════════════════
# Tests: Mask index selection
# ═══════════════════════════════════════════════════════════


class TestMaskIndexSelection:

    def test_correct_count(self, predictor):
        gd = get_gradients_by_prob(
            predictor, fake_ids(seq_len=10), 0, torch.randn(1, 3, 32, 32)
        )
        _, mag = get_gradient_magnitudes(gd, "normal_l1")
        indices = get_mask_indices(mag, mask_frac=0.3)
        assert len(indices) == math.ceil(0.3 * 10)

    def test_highest_first(self, predictor):
        gd = get_gradients_by_prob(
            predictor, fake_ids(seq_len=10), 0, torch.randn(1, 3, 32, 32)
        )
        _, mag = get_gradient_magnitudes(gd, "normal_l1")
        indices = get_mask_indices(mag, mask_frac=0.3)
        unmasked = [i for i in range(10) if i not in indices]
        assert mag[indices].min() >= mag[unmasked].max()

    def test_indices_within_text_range(self, predictor):
        """All mask indices should be valid text positions, not visual patch positions."""
        seq_len = 10
        gd = get_gradients_by_prob(
            predictor, fake_ids(seq_len=seq_len), 0, torch.randn(1, 3, 32, 32)
        )
        _, mag = get_gradient_magnitudes(gd, "normal_l1")
        indices = get_mask_indices(mag, mask_frac=0.5)
        for idx in indices:
            assert 0 <= idx < seq_len, f"Index {idx} outside text range [0, {seq_len})"

    def test_special_tokens_excluded(self, predictor):
        gd = get_gradients_by_prob(
            predictor, fake_ids(seq_len=10), 0, torch.randn(1, 3, 32, 32)
        )
        _, mag = get_gradient_magnitudes(gd, "normal_l1")
        specials = {0, 9}
        indices = get_mask_indices(mag, mask_frac=0.5, special_tok_indices=specials)
        assert all(i not in specials for i in indices)


# ═══════════════════════════════════════════════════════════
# Tests: Full pipeline — predictor saliency → mask → editor
# ═══════════════════════════════════════════════════════════


class TestFullPipeline:
    """
    End-to-end: compute saliency on multimodal predictor → mask text →
    feed masked text + images to multimodal editor → valid output.
    """

    def test_masked_text_through_editor(self, predictor, editor):
        input_ids = fake_ids(seq_len=10)
        images = torch.randn(1, 3, 32, 32)
        labels = fake_ids(seq_len=4)

        # Step 1: Get saliency from multimodal predictor
        grad_dict = get_gradients_by_prob(
            predictor, input_ids.clone(), pred_idx=0, images=images
        )
        _, magnitudes = get_gradient_magnitudes(grad_dict, "normal_l1")
        mask_indices = get_mask_indices(magnitudes, mask_frac=0.3)

        # Step 2: Mask text tokens
        masked_input = apply_sentinel_mask(input_ids, mask_indices)

        # Step 3: Feed masked text + same images to multimodal editor
        editor.eval()
        with torch.no_grad():
            out = editor(input_ids=masked_input, images=images, labels=labels)

        assert not torch.isnan(out["loss"])
        assert not torch.isinf(out["loss"])

    def test_masking_changes_editor_loss(self, predictor, editor):
        input_ids = fake_ids(seq_len=10)
        images = torch.randn(1, 3, 32, 32)
        labels = fake_ids(seq_len=4)

        grad_dict = get_gradients_by_prob(
            predictor, input_ids.clone(), pred_idx=0, images=images
        )
        _, magnitudes = get_gradient_magnitudes(grad_dict, "normal_l1")
        mask_indices = get_mask_indices(magnitudes, mask_frac=0.5)
        masked_input = apply_sentinel_mask(input_ids, mask_indices)

        editor.eval()
        with torch.no_grad():
            loss_orig = editor(input_ids=input_ids, images=images, labels=labels)[
                "loss"
            ]
            loss_masked = editor(input_ids=masked_input, images=images, labels=labels)[
                "loss"
            ]

        assert not torch.isclose(
            loss_orig, loss_masked, atol=1e-6
        ), "Masking had no effect on editor loss"

    def test_only_text_tokens_changed(self, predictor, editor):
        """After masking, only the selected text positions should differ."""
        input_ids = fake_ids(seq_len=10)
        images = torch.randn(1, 3, 32, 32)

        grad_dict = get_gradients_by_prob(
            predictor, input_ids.clone(), pred_idx=0, images=images
        )
        _, magnitudes = get_gradient_magnitudes(grad_dict, "normal_l1")
        mask_indices = get_mask_indices(magnitudes, mask_frac=0.3)
        masked_input = apply_sentinel_mask(input_ids, mask_indices)

        n_changed = (masked_input != input_ids).sum().item()
        assert n_changed == len(mask_indices)

        # Unmasked positions are identical
        unmask = torch.ones(10, dtype=torch.bool)
        for idx in mask_indices:
            unmask[idx] = False
        assert (masked_input[0, unmask] == input_ids[0, unmask]).all()

    @pytest.mark.parametrize("mask_frac", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_various_mask_fractions(self, predictor, editor, mask_frac):
        input_ids = fake_ids(seq_len=10)
        images = torch.randn(1, 3, 32, 32)
        labels = fake_ids(seq_len=4)

        grad_dict = get_gradients_by_prob(
            predictor, input_ids.clone(), pred_idx=0, images=images
        )
        _, magnitudes = get_gradient_magnitudes(grad_dict, "normal_l1")
        mask_indices = get_mask_indices(magnitudes, mask_frac=mask_frac)
        masked_input = apply_sentinel_mask(input_ids, mask_indices)

        editor.eval()
        with torch.no_grad():
            out = editor(input_ids=masked_input, images=images, labels=labels)
        assert not torch.isnan(out["loss"]), f"NaN at mask_frac={mask_frac}"

    def test_editor_text_only_still_works(self, predictor, editor):
        """If predictor is multimodal but editor gets no images, it should still work."""
        input_ids = fake_ids(seq_len=10)
        images = torch.randn(1, 3, 32, 32)
        labels = fake_ids(seq_len=4)

        # Saliency from multimodal predictor (with images)
        grad_dict = get_gradients_by_prob(
            predictor, input_ids.clone(), pred_idx=0, images=images
        )
        _, magnitudes = get_gradient_magnitudes(grad_dict, "normal_l1")
        mask_indices = get_mask_indices(magnitudes, mask_frac=0.3)
        masked_input = apply_sentinel_mask(input_ids, mask_indices)

        # Editor receives masked text WITHOUT images
        editor.eval()
        with torch.no_grad():
            out = editor(input_ids=masked_input, images=None, labels=labels)
        assert not torch.isnan(out["loss"])

    def test_batch_pipeline(self, predictor, editor):
        """Full pipeline with batch_size > 1."""
        batch = 2
        input_ids = fake_ids(batch=batch, seq_len=10)
        images = torch.randn(batch, 3, 32, 32)
        labels = fake_ids(batch=batch, seq_len=4)

        # Get saliency per sample (GradientMasker processes one at a time)
        masked_batch = input_ids.clone()
        for b in range(batch):
            gd = get_gradients_by_prob(
                predictor,
                input_ids[b : b + 1],
                pred_idx=0,
                images=images[b : b + 1],
            )
            _, mag = get_gradient_magnitudes(gd, "normal_l1")
            indices = get_mask_indices(mag, mask_frac=0.3)
            for idx in indices:
                masked_batch[b, idx] = 1

        editor.eval()
        with torch.no_grad():
            out = editor(input_ids=masked_batch, images=images, labels=labels)
        assert out["logits"].shape[0] == batch
        assert not torch.isnan(out["loss"])
