import torch
import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
from ..convert import convert


class MBartDecoderAttention(nn.Module):
    def __init__(self, hf_module):
        super().__init__()
        self.hf_module = hf_module

        self.num_heads = self.hf_module.num_heads
        self.head_dim = self.hf_module.head_dim
        self.embed_dim = self.hf_module.embed_dim
        self.scaling = self.hf_module.scaling

        self.q_proj = convert(self.hf_module.q_proj)
        self.k_proj = convert(self.hf_module.k_proj)
        self.v_proj = convert(self.hf_module.v_proj)

        self.out_proj = convert(self.hf_module.out_proj)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # hardcode as bs=1, seqlen=1
        return tensor.reshape(bsz, seq_len, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = mx.concatenate([past_key_value[0], key_states], axis=2)
            value_states = mx.concatenate([past_key_value[1], value_states], axis=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        past_key_value = (key_states, value_states)

        query_states = self._shape(query_states, tgt_len, bsz)

        attn_output = mx.fast.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=self.scaling
        )

        attn_output = attn_output.transpose(0, 2, 1, 3)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, past_key_value
