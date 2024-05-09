from time import time
from utils import get_model_and_processor, extract_pdf_as_image


model, processor = get_model_and_processor("facebook/nougat-base")
image = extract_pdf_as_image('1706.03762v7.pdf', 1)
pixel_values = processor(image, return_tensors="pt").pixel_values

# somehow running in bfloat16 on cpu will make it slower...
# import torch
# model.to(torch.bfloat16)
# pixel_values = pixel_values.to(torch.bfloat16)

print("start generation")
start_time = time()
# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values,
    min_length=1,
    max_new_tokens=4096,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)
sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
end_time = time()

print(sequence)
print(f"time takes: {end_time - start_time}")
