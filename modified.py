from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from time import time
from tqdm import tqdm

from utils import extract_pdf_as_image
from decode import decode


image = extract_pdf_as_image('1706.03762v7.pdf', 1)

processor = NougatProcessor.from_pretrained("facebook/nougat-base")

# prepare PDF image for the model
pixel_values = processor(image, return_tensors="pt").pixel_values

model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
model.eval()

print("start generation")

MAX_OUTPUT_LEN = 4096

with torch.no_grad():
    encoder_outputs = model.encoder(pixel_values)
    encoder_hidden_states = encoder_outputs[0]

    decoder_input_ids = torch.tensor([[0]])
    attention_mask = torch.ones((1, MAX_OUTPUT_LEN))
    outputs = [0]
    past_key_values = None
    for i in tqdm(range(MAX_OUTPUT_LEN)):
        logits, past_key_values = decode(
            model.decoder,
            input_ids=decoder_input_ids,
            attention_mask=attention_mask[:, :i + 1],
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )
        # greedy sampling
        new_token = torch.argmax(logits).item()
        if new_token == 2:
            break
        decoder_input_ids = torch.tensor([[new_token]])
        outputs.append(new_token)


end_time = time()

outputs = torch.tensor([outputs])
sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)

print(sequence)
