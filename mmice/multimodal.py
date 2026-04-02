import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from typing import Optional, Tuple, List, Dict
import copy
from transformers.models.t5.modeling_t5 import (
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5LayerFF,
)


############# Vision Encoder #################


class VisionEncoder(nn.Module):
    """Vision Transformer for encoding images as patch sequences"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim) * 0.02
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x[:, 1:, :]  # Remove CLS token


class TransformerBlock(nn.Module):
    """Standard transformer block"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


########### Perceived Resampler (Forked from flamingo) ###########################


class PerceiverResampler(nn.Module):
    """
    Flamingo-style Perceiver Resampler
    Compresses variable-length visual features into fixed number of tokens
    """

    def __init__(
        self,
        dim: int,
        depth: int = 6,
        num_latents: int = 64,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_latents = num_latents

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # Perceiver layers
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverCrossAttention(dim, num_heads, dropout),
                        PerceiverSelfAttention(dim, num_heads, dropout),
                        FeedForward(dim, ff_mult, dropout),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Visual features (batch, num_patches, dim)
        Returns:
            Compressed features (batch, num_latents, dim)
        """
        batch_size = x.shape[0]

        # Expand learned latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply perceiver layers
        for cross_attn, self_attn, ff in self.layers:
            latents = cross_attn(latents, x) + latents
            latents = self_attn(latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


class PerceiverCrossAttention(nn.Module):
    """Cross-attention where queries are latents, keys/values are visual features"""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latents_norm = self.norm_latents(latents)
        context_norm = self.norm_context(context)

        out, _ = self.attn(
            query=latents_norm,
            key=context_norm,
            value=context_norm,
        )
        return out


class PerceiverSelfAttention(nn.Module):
    """Self-attention among latent tokens"""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm)
        return out


class FeedForward(nn.Module):
    """Feed-forward network"""

    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


################ Gated Cross-attention ###################


class GatedCrossAttentionBlock(nn.Module):
    """
    Gated cross-attention that allows decoder to attend to visual features
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.norm_hidden = nn.LayerNorm(dim)
        self.norm_visual = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Gating parameter (starts near 0)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_norm = self.norm_hidden(hidden_states)
        visual_norm = self.norm_visual(visual_features)

        # Cross-attention
        attn_output, _ = self.cross_attn(
            query=hidden_norm,
            key=visual_norm,
            value=visual_norm,
            key_padding_mask=attention_mask,
        )

        # Apply gating
        gate = torch.tanh(self.alpha)

        # Gated residual
        return hidden_states + gate * attn_output


######### T5 Extensions ###############


class T5DecoderLayerWithVision(nn.Module):
    """
    Enhanced T5 decoder layer with visual cross-attention

    Order of operations:
    1. Self-attention (causal)
    2. Cross-attention to encoder outputs
    3. Gated cross-attention to visual features [NEW]
    4. Feed-forward
    """

    def __init__(
        self,
        config: T5Config,
        has_visual_attention: bool = True,
    ):
        super().__init__()
        self.is_decoder = True
        self.has_visual_attention = has_visual_attention

        # Standard T5 decoder components
        self.layer = nn.ModuleList()

        # Self-attention
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=False)
        )

        # Cross-attention to encoder
        self.layer.append(T5LayerCrossAttention(config))

        # Feed-forward
        self.layer.append(T5LayerFF(config))

        # Visual cross-attention (NEW)
        if has_visual_attention:
            self.visual_cross_attention = GatedCrossAttentionBlock(
                dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout_rate,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_decoder_position_bias: Optional[torch.Tensor] = None,
        visual_features: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple:
        """Forward pass with visual cross-attention"""

        # Self-attention
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]

        # Cross-attention to encoder
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

        # Visual cross-attention (NEW!)
        if self.has_visual_attention and visual_features is not None:
            hidden_states = self.visual_cross_attention(
                hidden_states=hidden_states,
                visual_features=visual_features,
            )

        # Feed-forward
        hidden_states = self.layer[2](hidden_states)

        outputs = (hidden_states,)

        return outputs


class T5DecoderBlockWrapper(nn.Module):
    """
    Wrapper around T5 decoder block that adds visual cross-attention
    WITHOUT recreating the original layers
    """

    def __init__(
        self,
        original_block: nn.Module,
        config: T5Config,
        has_visual_attention: bool = True,
    ):
        super().__init__()
        # Keep original block as-is
        self.original_block = original_block
        self.has_visual_attention = has_visual_attention

        # Add visual cross-attention
        if has_visual_attention:
            self.visual_cross_attention = GatedCrossAttentionBlock(
                dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout_rate,
            )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        visual_features=None,
        **kwargs,
    ):
        """Forward pass with optional visual cross-attention"""

        # Get the signature of the original block's forward method
        import inspect

        sig = inspect.signature(self.original_block.forward)
        valid_params = set(sig.parameters.keys())

        # Build kwargs for original block (only include supported args)
        forward_kwargs = {}

        # Always pass positional arg
        all_args = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "position_bias": position_bias,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "encoder_decoder_position_bias": encoder_decoder_position_bias,
            "layer_head_mask": layer_head_mask,
            "cross_attn_layer_head_mask": cross_attn_layer_head_mask,
            "past_key_value": past_key_value,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
        }

        # Only include args that the original block accepts
        for key, value in all_args.items():
            if key in valid_params and value is not None:
                forward_kwargs[key] = value
            elif key in valid_params:
                forward_kwargs[key] = value

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key in valid_params:
                forward_kwargs[key] = value

        # Call original T5 decoder block
        outputs = self.original_block(**forward_kwargs)

        # Extract hidden states
        hidden_states = outputs[0]

        # Apply visual cross-attention if available
        if self.has_visual_attention:
            # Check if visual features are cached in this layer
            if hasattr(self, "_current_visual_features"):
                hidden_states = self.visual_cross_attention(
                    hidden_states=hidden_states,
                    visual_features=self._current_visual_features,
                )
            elif visual_features is not None:
                # Use visual features passed as argument
                hidden_states = self.visual_cross_attention(
                    hidden_states=hidden_states,
                    visual_features=visual_features,
                )

        # Return in same format as original
        return (hidden_states,) + outputs[1:]


class MultimodalT5ForConditionalGeneration(nn.Module):
    """
    Multimodal T5 for conditional generation with TRAINABLE LM layers

    Use cases:
    - Image captioning: image -> caption
    - VQA: image + question -> answer
    - Image-to-text translation
    - Multimodal summarization
    """

    def __init__(
        self,
        t5_model_name: str = "t5-base",
        vision_config: dict = None,
        use_perceiver: bool = True,
        num_perceiver_latents: int = 64,
        perceiver_depth: int = 6,
        decoder_cross_attn_layers: List[int] = None,
        freeze_vision: bool = False,  # Can optionally freeze vision encoder
        freeze_t5: bool = False,  # Set to False for trainable T5!
    ):
        super().__init__()

        # Load T5
        self.config = T5Config.from_pretrained(t5_model_name)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)

        # Vision encoder
        vision_config = vision_config or {}
        vision_embed_dim = vision_config.get("embed_dim", 768)

        self.vision_encoder = VisionEncoder(
            img_size=vision_config.get("img_size", 224),
            patch_size=vision_config.get("patch_size", 16),
            embed_dim=vision_embed_dim,
            depth=vision_config.get("depth", 12),
            num_heads=vision_config.get("num_heads", 12),
        )

        # Perceiver resampler
        self.use_perceiver = use_perceiver
        if use_perceiver:
            self.perceiver = PerceiverResampler(
                dim=vision_embed_dim,
                depth=perceiver_depth,
                num_latents=num_perceiver_latents,
                num_heads=8,
            )

        # Projection to T5 dimension
        hidden_size = self.config.d_model
        if vision_embed_dim != hidden_size:
            self.visual_projection = nn.Linear(vision_embed_dim, hidden_size)
        else:
            self.visual_projection = nn.Identity()

        # Determine which decoder layers get visual cross-attention
        if decoder_cross_attn_layers is None:
            # Default: add to every 3rd layer
            total_layers = self.config.num_decoder_layers
            decoder_cross_attn_layers = list(range(0, total_layers, 3))

        self.decoder_cross_attn_layers = decoder_cross_attn_layers

        # Wrap decoder layers
        self._wrap_decoder_layers()

        # Optionally freeze components
        if freeze_vision:
            self._freeze_vision_encoder()

        if freeze_t5:
            self._freeze_t5()
        else:
            print("T5 layers are TRAINABLE (unfrozen)")

    def _wrap_decoder_layers(self):
        """Wrap decoder layers to add visual cross-attention"""
        print("Wrapping decoder layers...")
        new_decoder_layers = nn.ModuleList()

        for idx, layer in enumerate(self.t5.decoder.block):
            if idx in self.decoder_cross_attn_layers:
                # Wrap with visual cross-attention
                wrapped_layer = T5DecoderBlockWrapper(
                    original_block=layer,
                    config=self.config,
                    has_visual_attention=True,
                )
                new_decoder_layers.append(wrapped_layer)
                print(f"\tLayer {idx}: Added visual cross-attention")
            else:
                # Wrap without visual cross-attention
                wrapped_layer = T5DecoderBlockWrapper(
                    original_block=layer,
                    config=self.config,
                    has_visual_attention=False,
                )
                new_decoder_layers.append(wrapped_layer)

        # Replace decoder layers
        self.t5.decoder.block = new_decoder_layers
        print(f"Wrapped {len(new_decoder_layers)} decoder layers")

    def _freeze_vision_encoder(self):
        """Freeze vision encoder parameters"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")

    def _freeze_t5(self):
        """Freeze T5 parameters (typically not used in this version!)"""
        for param in self.t5.parameters():
            param.requires_grad = False
        # Unfreeze visual cross-attention
        for layer in self.t5.decoder.block:
            if hasattr(layer, "visual_cross_attention"):
                for param in layer.visual_cross_attention.parameters():
                    param.requires_grad = True
        print("T5 layers frozen (except visual cross-attention)")

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to visual features

        Args:
            images: (batch_size, channels, height, width)
        Returns:
            visual_features: (batch_size, num_visual_tokens, hidden_size)
        """
        # Vision encoder
        visual_features = self.vision_encoder(images)

        # Optional perceiver resampler
        if self.use_perceiver:
            visual_features = self.perceiver(visual_features)

        # Project to T5 dimension
        visual_features = self.visual_projection(visual_features)

        return visual_features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""

        # Encode images
        visual_features = None
        if images is not None:
            visual_features = self.encode_images(images)

        # Prepare decoder input
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self.t5._shift_right(labels)

        # Store visual features for decoder layers to access
        # Each wrapped layer will check for this attribute
        if visual_features is not None:
            for layer in self.t5.decoder.block:
                if isinstance(layer, T5DecoderBlockWrapper):
                    layer._current_visual_features = visual_features

        try:
            # Forward through T5
            outputs = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs,
            )
        finally:
            # Clean up visual features from layers
            if visual_features is not None:
                for layer in self.t5.decoder.block:
                    if isinstance(layer, T5DecoderBlockWrapper) and hasattr(
                        layer, "_current_visual_features"
                    ):
                        delattr(layer, "_current_visual_features")

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_length: int = 50,
        **kwargs,
    ):
        """Generate text"""
        # Encode images
        if images is not None:
            visual_features = self.encode_images(images)
            # Store in decoder layers
            for layer in self.t5.decoder.block:
                if isinstance(layer, T5DecoderBlockWrapper):
                    layer._current_visual_features = visual_features

        try:
            outputs = self.t5.generate(
                input_ids=input_ids, max_length=max_length, **kwargs
            )
        finally:
            # Clean up
            if images is not None:
                for layer in self.t5.decoder.block:
                    if isinstance(layer, T5DecoderBlockWrapper) and hasattr(
                        layer, "_current_visual_features"
                    ):
                        delattr(layer, "_current_visual_features")

        return outputs
