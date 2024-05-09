from time import time
from utils import get_model_and_processor, extract_pdf_as_image
from faster_nougat import generate

model, processor = get_model_and_processor("facebook/nougat-small")

image = extract_pdf_as_image('1706.03762v7.pdf', 1)
pixel_values = processor(image, return_tensors="pt").pixel_values

print("start generation")
start_time = time()
outputs = generate(model, pixel_values, max_new_tokens=4096)
sequence = processor.batch_decode([outputs], skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
end_time = time()

print(sequence)
print(f"time takes: {end_time - start_time}")
