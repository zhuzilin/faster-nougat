import torch
from transformers import MBartForCausalLM

def decode(
    model: MBartForCausalLM,
    *,
    input_ids,
    attention_mask,
    encoder_hidden_states,
    past_key_values,
):
    decoder_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
        use_cache=True,
        past_key_values=past_key_values,
        return_dict=True,
    )



    logits = decoder_output['logits'].flatten()
    past_key_values = decoder_output['past_key_values']
    return logits, past_key_values
