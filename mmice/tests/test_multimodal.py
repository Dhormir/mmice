"""
Sanity checks for multimodal.py

Run:
    uv run --with torch --with transformers --with sentencepiece --with protobuf --with pytest \
        pytest test_multimodal.py -v
"""

import pytest
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config

from mmice.multimodal import (
    VisionEncoder,
    TransformerBlock,
    PerceiverResampler,
    FeedForward,
    GatedCrossAttentionBlock,
    T5DecoderBlockWrapper,
    MultimodalT5ForConditionalGeneration,
)


# ─── Fixtures ───


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
D_MODEL = TINY_T5_CONFIG.d_model  # 64


@pytest.fixture
def tiny_model():
    """Build a MultimodalT5 from a tiny random T5 (no download)."""
    return _make_tiny_model()


def _make_tiny_model(**overrides):
    kwargs = dict(
        t5_model_name="__tiny__",
        vision_config=TINY_VIS,
        use_perceiver=True,
        num_perceiver_latents=4,
        perceiver_depth=1,
    )
    kwargs.update(overrides)

    _orig_init = MultimodalT5ForConditionalGeneration.__init__

    def _patched_init(self, **kw):
        nn.Module.__init__(self)
        self.config = TINY_T5_CONFIG
        self.t5 = T5ForConditionalGeneration(TINY_T5_CONFIG)

        vision_config = kw.get("vision_config", {})
        vision_embed_dim = vision_config.get("embed_dim", 768)

        self.vision_encoder = VisionEncoder(
            img_size=vision_config.get("img_size", 224),
            patch_size=vision_config.get("patch_size", 16),
            embed_dim=vision_embed_dim,
            depth=vision_config.get("depth", 12),
            num_heads=vision_config.get("num_heads", 12),
        )

        self.use_perceiver = kw.get("use_perceiver", True)
        if self.use_perceiver:
            self.perceiver = PerceiverResampler(
                dim=vision_embed_dim,
                depth=kw.get("perceiver_depth", 6),
                num_latents=kw.get("num_perceiver_latents", 64),
                num_heads=4,
            )

        hidden_size = self.config.d_model
        if vision_embed_dim != hidden_size:
            self.visual_projection = nn.Linear(vision_embed_dim, hidden_size)
        else:
            self.visual_projection = nn.Identity()

        dcal = kw.get("decoder_cross_attn_layers", None)
        if dcal is None:
            total_layers = self.config.num_decoder_layers
            dcal = list(range(0, total_layers, 3))
        self.decoder_cross_attn_layers = dcal

        self._wrap_decoder_layers()

        if kw.get("freeze_vision", False):
            self._freeze_vision_encoder()
        if kw.get("freeze_t5", False):
            self._freeze_t5()

    MultimodalT5ForConditionalGeneration.__init__ = lambda self, **kw: _patched_init(
        self, **kw
    )
    try:
        model = MultimodalT5ForConditionalGeneration(**kwargs)
    finally:
        MultimodalT5ForConditionalGeneration.__init__ = _orig_init
    return model


def fake_ids(batch=1, seq_len=5):
    return torch.randint(2, TINY_T5_CONFIG.vocab_size, (batch, seq_len))


def _enable_generate(model):
    """Set generation config fields needed for tiny model."""
    model.t5.config.decoder_start_token_id = 0
    model.t5.generation_config.decoder_start_token_id = 0
    model.t5.generation_config.eos_token_id = 1
    model.t5.generation_config.pad_token_id = 0


# ─── Component Tests ───


class TestVisionEncoder:
    def test_output_shape(self):
        enc = VisionEncoder(
            img_size=64, patch_size=16, embed_dim=128, depth=2, num_heads=4
        )
        out = enc(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 16, 128)

    @pytest.mark.parametrize("img_size,patch_size", [(32, 8), (96, 16), (128, 32)])
    def test_different_image_sizes(self, img_size, patch_size):
        enc = VisionEncoder(
            img_size=img_size, patch_size=patch_size, embed_dim=64, depth=1, num_heads=4
        )
        out = enc(torch.randn(1, 3, img_size, img_size))
        expected_patches = (img_size // patch_size) ** 2
        assert out.shape == (1, expected_patches, 64)

    def test_cls_token_removed(self):
        enc = VisionEncoder(
            img_size=64, patch_size=16, embed_dim=64, depth=1, num_heads=4
        )
        out = enc(torch.randn(1, 3, 64, 64))
        assert out.shape[1] == (64 // 16) ** 2  # no +1 for CLS


class TestTransformerBlock:
    def test_residual_shape(self):
        block = TransformerBlock(dim=64, num_heads=4, mlp_ratio=4.0, dropout=0.0)
        x = torch.randn(2, 10, 64)
        assert block(x).shape == x.shape


class TestPerceiverResampler:
    def test_compresses_to_num_latents(self):
        pr = PerceiverResampler(dim=128, depth=2, num_latents=8, num_heads=4)
        out = pr(torch.randn(2, 196, 128))
        assert out.shape == (2, 8, 128)

    @pytest.mark.parametrize("seq_len", [16, 64, 256])
    def test_variable_input_lengths(self, seq_len):
        pr = PerceiverResampler(dim=64, depth=1, num_latents=4, num_heads=4)
        out = pr(torch.randn(1, seq_len, 64))
        assert out.shape == (1, 4, 64)


class TestGatedCrossAttention:
    def test_gate_init_zero(self):
        gca = GatedCrossAttentionBlock(dim=64, num_heads=4)
        assert gca.alpha.item() == 0.0

    def test_near_identity_at_init(self):
        gca = GatedCrossAttentionBlock(dim=64, num_heads=4, dropout=0.0)
        hidden = torch.randn(2, 10, 64)
        visual = torch.randn(2, 8, 64)
        out = gca(hidden, visual)
        assert out.shape == hidden.shape
        assert (out - hidden).abs().max().item() < 1e-5


class TestFeedForward:
    def test_shape_preservation(self):
        ff = FeedForward(dim=64, mult=4, dropout=0.0)
        x = torch.randn(2, 10, 64)
        assert ff(x).shape == x.shape


# ─── Integration Tests ───


class TestT5DecoderBlockWrapper:
    def test_preserves_output_format(self):
        base_model = T5ForConditionalGeneration(TINY_T5_CONFIG)
        wrapper = T5DecoderBlockWrapper(
            base_model.decoder.block[0], TINY_T5_CONFIG, has_visual_attention=True
        )
        hidden = torch.randn(1, 5, D_MODEL)
        cache_position = torch.arange(5)
        assert (
            wrapper(hidden_states=hidden, cache_position=cache_position)[0].shape
            == hidden.shape
        )


class TestMultimodalT5Forward:
    def test_with_images_produces_loss(self, tiny_model):
        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(
                input_ids=fake_ids(),
                images=torch.randn(1, 3, 32, 32),
                labels=fake_ids(seq_len=3),
            )
        assert out["loss"].dim() == 0
        assert not torch.isnan(out["loss"])
        assert not torch.isinf(out["loss"])

    def test_without_images_fallback(self, tiny_model):
        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(
                input_ids=fake_ids(), images=None, labels=fake_ids(seq_len=3)
            )
        assert not torch.isnan(out["loss"])

    def test_batch_size_gt_1(self, tiny_model):
        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(
                input_ids=fake_ids(batch=2),
                images=torch.randn(2, 3, 32, 32),
                labels=fake_ids(batch=2, seq_len=3),
            )
        assert out["logits"].shape[0] == 2

    def test_output_keys(self, tiny_model):
        tiny_model.eval()
        with torch.no_grad():
            out = tiny_model(
                input_ids=fake_ids(),
                images=torch.randn(1, 3, 32, 32),
                labels=fake_ids(seq_len=3),
            )
        assert "loss" in out
        assert "logits" in out


class TestMultimodalT5Cleanup:
    def test_visual_features_cleaned_after_forward(self, tiny_model):
        tiny_model.eval()
        with torch.no_grad():
            tiny_model(
                input_ids=fake_ids(),
                images=torch.randn(1, 3, 32, 32),
                labels=fake_ids(seq_len=3),
            )
        for i, layer in enumerate(tiny_model.t5.decoder.block):
            assert not hasattr(layer, "_current_visual_features"), f"Layer {i} leaked"

    def test_visual_features_cleaned_on_error(self, tiny_model):
        try:
            tiny_model(
                input_ids=torch.tensor([[99999]]),
                images=torch.randn(1, 3, 32, 32),
                labels=torch.tensor([[0]]),
            )
        except Exception:
            pass
        for i, layer in enumerate(tiny_model.t5.decoder.block):
            assert not hasattr(
                layer, "_current_visual_features"
            ), f"Layer {i} leaked after error"


class TestMultimodalT5Generate:
    def test_with_images(self, tiny_model):
        tiny_model.eval()
        _enable_generate(tiny_model)
        with torch.no_grad():
            out = tiny_model.generate(
                input_ids=fake_ids(), images=torch.randn(1, 3, 32, 32), max_length=10
            )
        assert out.dim() == 2
        assert out.shape[0] == 1
        assert out.shape[1] > 0

    def test_without_images(self, tiny_model):
        tiny_model.eval()
        _enable_generate(tiny_model)
        with torch.no_grad():
            out = tiny_model.generate(input_ids=fake_ids(), images=None, max_length=10)
        assert out.shape[0] == 1


class TestMultimodalT5Backward:
    def test_gradients_flow(self):
        model = _make_tiny_model(freeze_t5=False, freeze_vision=False)
        model.train()
        out = model(
            input_ids=fake_ids(),
            images=torch.randn(1, 3, 32, 32),
            labels=fake_ids(seq_len=3),
        )
        out["loss"].backward()

        assert model.vision_encoder.patch_embed.weight.grad is not None
        assert model.perceiver.latents.grad is not None

        found_grad = any(
            hasattr(layer, "visual_cross_attention")
            and layer.visual_cross_attention.alpha.grad is not None
            for layer in model.t5.decoder.block
        )
        assert found_grad, "No gradient on any gated cross-attention alpha"


class TestMultimodalT5Freeze:
    def test_freeze_vision(self):
        model = _make_tiny_model(freeze_vision=True)
        for name, p in model.vision_encoder.named_parameters():
            assert not p.requires_grad, f"{name} should be frozen"

    def test_freeze_t5_keeps_visual_crossattn_trainable(self):
        model = _make_tiny_model(freeze_t5=True)
        assert sum(1 for p in model.t5.parameters() if not p.requires_grad) > 0
        for layer in model.t5.decoder.block:
            if hasattr(layer, "visual_cross_attention"):
                for name, p in layer.visual_cross_attention.named_parameters():
                    assert p.requires_grad, f"{name} should be trainable"


class TestMultimodalT5Config:
    def test_decoder_cross_attn_layers_respected(self):
        model = _make_tiny_model(decoder_cross_attn_layers=[0, 3, 5])
        for i, layer in enumerate(model.t5.decoder.block):
            has_vis = hasattr(layer, "visual_cross_attention")
            if i in [0, 3, 5]:
                assert has_vis, f"Layer {i} missing visual cross-attention"
            else:
                assert not has_vis, f"Layer {i} shouldn't have visual cross-attention"

    def test_more_crossattn_layers_means_more_params(self):
        model_few = _make_tiny_model(decoder_cross_attn_layers=[0])
        model_all = _make_tiny_model(decoder_cross_attn_layers=[0, 1, 2, 3, 4, 5])
        params_few = sum(p.numel() for p in model_few.parameters())
        params_all = sum(p.numel() for p in model_all.parameters())
        assert params_all > params_few


class TestEncodeImages:
    def test_with_perceiver(self):
        model = _make_tiny_model(use_perceiver=True, num_perceiver_latents=8)
        feats = model.encode_images(torch.randn(2, 3, 32, 32))
        assert feats.shape == (2, 8, D_MODEL)

    def test_without_perceiver(self):
        model = _make_tiny_model(use_perceiver=False)
        feats = model.encode_images(torch.randn(2, 3, 32, 32))
        num_patches = (32 // 8) ** 2
        assert feats.shape == (2, num_patches, D_MODEL)
