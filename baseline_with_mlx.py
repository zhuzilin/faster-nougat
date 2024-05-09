import torch
import mlx.core as mx
from time import time
from tqdm import tqdm

from utils import get_model_and_processor, extract_pdf_as_image
from faster_nougat.layers.mbart_decode import MBartDecoder
from faster_nougat.convert import convert

model, processor = get_model_and_processor()

image = extract_pdf_as_image('1706.03762v7.pdf', 1)
pixel_values = processor(image, return_tensors="pt").pixel_values

print("start generation")
start_time = time()
with torch.no_grad():
    encoder_outputs = model.encoder(pixel_values)
    encoder_hidden_states = encoder_outputs[0]
    encoder_hidden_states = convert(encoder_hidden_states)

    decoder = MBartDecoder(model.decoder)
    decoder.freeze()
    new_token = 0
    outputs = [0]
    past_key_values = None
    for i in tqdm(range(4096)):
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


end_time = time()

outputs = torch.tensor([outputs])
sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)

print(sequence)
print(f"time takes: {end_time - start_time}")
