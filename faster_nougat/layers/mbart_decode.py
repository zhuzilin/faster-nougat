import torch
import mlx.nn as nn
from typing import Optional, Tuple
from .mbart_decoder_layer import MBartDecoderLayer
from ..convert import convert


class MBartDecoder(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.lm_head = convert(hf_model.lm_head)

        self.hf_model = hf_model.model.decoder
        # embedding
        self.embed_tokens = convert(self.hf_model.embed_tokens)
        self.embed_positions = convert(self.hf_model.embed_positions)
        self.layernorm_embedding = convert(self.hf_model.layernorm_embedding)

        # decode layers
        self.layers = [
            MBartDecoderLayer(decoder_layer) for decoder_layer in self.hf_model.layers
        ]

        # output
        self.layer_norm = convert(self.hf_model.layer_norm)
        self.embed_scale = self.hf_model.embed_scale

        self.decode_count = 0

    def emb(self, input_ids: int):
        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        positions = self.embed_positions(self.decode_count + 2)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        return hidden_states

    def __call__(
        self,
        input_ids: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> Tuple:
        hidden_states = self.emb(input_ids)
        hidden_states = hidden_states.reshape(1, 1, -1)

        # decoder layers
        next_decoder_cache = ()
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_value=past_key_value,
            )
            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        self.decode_count += 1
        return logits, next_decoder_cache
