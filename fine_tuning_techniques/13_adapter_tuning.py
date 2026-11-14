"""
13. Adapter-Based Fine-Tuning
==============================
Pioneered by Houlsby et al. (ICML 2019) and Pfeiffer et al. (EMNLP 2020).

Architecture & Mechanics:
- Classical bottleneck adapters inject small residual subnetworks after the Multi-Head
  Attention (MHA) and Feed-Forward Network (FFN) layers of each transformer block.
- Structure of a Bottleneck Adapter:
      x ────────────────────────┐ (residual skip connection)
      │                         │
      ▼                         │
   Down-Projection (d -> m)     │   where bottleneck dimension m << d
      │                         │
      ▼                         │
   Non-linear Activation (GeLU) │
      │                         │
      ▼                         │
   Up-Projection (m -> d)       │
      │                         │
      ▼                         ▼
   Output = LayerNorm( x + Up(GeLU(Down(x))) )

Parameters added: Typically 0.5% - 1% of total model size.
Base transformer weights remain 100% frozen during training.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class BottleneckAdapter(nn.Module):
    """
    Standard Houlsby/Pfeiffer Bottleneck Adapter layer with residual skip-connection.
    """
    def __init__(self, hidden_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Initialize up_proj to near-zero so initial adapter behaves as an identity function
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        adapter_out = self.down_proj(x)
        adapter_out = self.activation(adapter_out)
        adapter_out = self.dropout(adapter_out)
        adapter_out = self.up_proj(adapter_out)
        return self.layer_norm(residual + adapter_out)


class TransformerBlockWithAdapter(nn.Module):
    """Wraps a frozen transformer block with an additive bottleneck adapter."""
    def __init__(self, original_layer: nn.Module, hidden_dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.original_layer = original_layer
        self.adapter = BottleneckAdapter(hidden_dim, bottleneck_dim)

        # Freeze the underlying original layer
        for p in self.original_layer.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        layer_output = self.original_layer(*args, **kwargs)
        if isinstance(layer_output, tuple):
            hidden_states = layer_output[0]
            rest = layer_output[1:]
            adapted_states = self.adapter(hidden_states)
            return (adapted_states,) + rest
        else:
            return self.adapter(layer_output)


def attach_bottleneck_adapters(model: nn.Module, bottleneck_dim: int = 64):
    """Freezes base model and attaches bottleneck adapters to all transformer layers."""
    for param in model.parameters():
        param.requires_grad = False

    # Example: attach adapters to decoder layers
    hidden_dim = getattr(model.config, "hidden_size", 4096)
    layers = getattr(model.model, "layers", [])

    for i, layer in enumerate(layers):
        layers[i] = TransformerBlockWithAdapter(layer, hidden_dim=hidden_dim, bottleneck_dim=bottleneck_dim)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[Adapter Tuning] Trainable adapter parameters: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    return model


if __name__ == "__main__":
    print("Bottleneck Adapter (Houlsby / Pfeiffer) module initialized.")
    adapter = BottleneckAdapter(hidden_dim=4096, bottleneck_dim=64)
    dummy_input = torch.randn(2, 16, 4096)
    output = adapter(dummy_input)
    print(f"Input shape: {dummy_input.shape} -> Output shape: {output.shape} (Residual identity preserved)")
