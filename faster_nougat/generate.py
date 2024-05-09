import torch
import mlx.core as mx
from tqdm import tqdm
from .convert import convert
from .layers.mbart_decode import MBartDecoder


def generate(model, pixel_values, *, max_new_tokens=4096, disable_tqdm=False):
    with torch.no_grad():
        encoder_outputs = model.encoder(pixel_values)
        encoder_hidden_states = encoder_outputs[0]
        encoder_hidden_states = convert(encoder_hidden_states).astype(mx.bfloat16)

        decoder = MBartDecoder(model.decoder)
        decoder.eval()
        decoder.set_dtype(mx.bfloat16)
        new_token = 0
        outputs = [0]
        past_key_values = None
        for _ in tqdm(range(max_new_tokens), disable=disable_tqdm):
            logits, past_key_values = decoder(
                input_ids=new_token,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
            )
            # greedy sampling
            new_token = logits.argmax().item()
            if new_token == 2:
                break
            outputs.append(new_token)
    return outputs
