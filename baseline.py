from transformers import NougatProcessor, VisionEncoderDecoderModel
import torch
from time import time

from wand.image import Image as WandImage
import numpy as np
from PIL import Image as PILImage

# Load the image using wand
with WandImage(filename='1706.03762v7.pdf[1]', resolution=200) as wand_img:
    # Convert wand image to numpy array
    numpy_array = np.array(wand_img)
    # Create a new PIL Image object from the numpy array
    image = PILImage.fromarray(numpy_array).convert("RGB")

processor = NougatProcessor.from_pretrained("facebook/nougat-base")

# prepare PDF image for the model
pixel_values = processor(image, return_tensors="pt").pixel_values

model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")

print("start generation")
start_time = time()
# generate transcription (here we only generate 30 tokens)
outputs = model.generate(
    pixel_values,
    min_length=1,
    max_new_tokens=4096,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)
end_time = time()

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)

print(f"time takes: {end_time - start_time}")
print(sequence)
