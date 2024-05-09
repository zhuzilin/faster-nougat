import numpy as np
from time import time
from PIL import Image as PILImage
from wand.image import Image as WandImage
from transformers import NougatProcessor, VisionEncoderDecoderModel


def get_model_and_processor(model_name):
    print("start loading model and processor")
    start = time()
    processor = NougatProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.eval()
    end = time()
    print(f"time: {end - start:.2f} s")
    return model, processor


def extract_pdf_as_image(filename: str, page_idx: int):
    assert filename.endswith(".pdf"), f"{filename} does not end with .pdf, is it a pdf file?"
    # Load the image using wand
    with WandImage(filename=f'{filename}[{page_idx}]', resolution=200) as wand_img:
        # Convert wand image to numpy array
        numpy_array = np.array(wand_img)
        # Create a new PIL Image object from the numpy array
        image = PILImage.fromarray(numpy_array).convert("RGB")
    return image
