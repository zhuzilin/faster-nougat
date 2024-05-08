import torch
import mlx.core as mx
from time import time
from tqdm import tqdm

from utils import get_model_and_processor, extract_pdf_as_image
from faster_nougat.decode import decode, Decoder
from faster_nougat.convert import convert


model, processor = get_model_and_processor()
image = extract_pdf_as_image('1706.03762v7.pdf', 1)
pixel_values = processor(image, return_tensors="pt").pixel_values

print("start generation")
start_time = time()
MAX_OUTPUT_LEN = 4096
with torch.no_grad():
    encoder_outputs = model.encoder(pixel_values)
    encoder_hidden_states = encoder_outputs[0]
    encoder_hidden_states = convert(encoder_hidden_states)

    decoder = Decoder(model.decoder)
    decoder_input_ids = 0
    attention_mask = mx.zeros((1, 1, 1, MAX_OUTPUT_LEN))
    outputs = [0]
    past_key_values = None
    for i in tqdm(range(MAX_OUTPUT_LEN)):
        logits, past_key_values = decode(
            decoder,
            input_ids=decoder_input_ids,
            attention_mask=attention_mask[:, :, :, :i + 1],
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )
        # greedy sampling
        new_token = logits.argmax().item()
        if new_token == 2:
            break
        decoder_input_ids = new_token
        outputs.append(new_token)


end_time = time()

outputs = torch.tensor([outputs])
sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)

print(sequence)
print(f"time takes: {end_time - start_time}")
