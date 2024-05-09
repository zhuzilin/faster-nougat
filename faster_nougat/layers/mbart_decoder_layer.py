import torch
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from ..convert import convert
from .mbart_attention import MBartDecoderAttention


class MBartDecoderLayer(nn.Module):
    def __init__(self, hf_module):
        super().__init__()
        self.hf_module = hf_module

        self.self_attn_layer_norm = convert(self.hf_module.self_attn_layer_norm)
        self.encoder_attn_layer_norm = convert(self.hf_module.encoder_attn_layer_norm)

        self.self_attn = MBartDecoderAttention(self.hf_module.self_attn)
        self.encoder_attn = MBartDecoderAttention(self.hf_module.encoder_attn)

        self.final_layer_norm = convert(self.hf_module.final_layer_norm)
        self.fc1 = convert(self.hf_module.fc1)
        # hardcode gelu. should be fine...
        self.activation_fn = nn.gelu
        self.fc2 = convert(self.hf_module.fc2)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = (
            past_key_value[:2] if past_key_value is not None else None
        )
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
        )

        hidden_states = residual + hidden_states

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
        cross_attn_past_key_value = (
            past_key_value[-2:] if past_key_value is not None else None
        )
        hidden_states, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            past_key_value=cross_attn_past_key_value,
        )

        hidden_states = residual + hidden_states

        # add cross-attn to positions 3,4 of present_key_value tuple
        present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value
