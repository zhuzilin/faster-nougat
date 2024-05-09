import numpy as np
from time import time
from PIL import Image as PILImage
from wand.image import Image as WandImage
from wand.color import Color
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


def extract_single_pdf_page_as_image(filename: str, page_idx: int, *, resolution=200):
    assert filename.endswith(
        ".pdf"
    ), f"{filename} does not end with .pdf, is it a pdf file?"
    assert page_idx > 0, f"page_idx starts from 1 to align with most pdf readers"
    with WandImage(
        filename=f"{filename}[{page_idx - 1}]", resolution=resolution
    ) as wand_img:
        wand_img.background_color = Color("white")
        wand_img.alpha_channel = "remove"
        numpy_array = np.array(wand_img)
        image = PILImage.fromarray(numpy_array).convert("RGB")
    return image


def extract_pdf_pages_as_images(filename: str, *, resolution=200):
    assert filename.endswith(
        ".pdf"
    ), f"{filename} does not end with .pdf, is it a pdf file?"
    print(f"start extracting {filename}")
    # Load the image using wand
    images = []
    with WandImage(filename=filename, resolution=resolution) as wand_imgs:
        print(f"the pdf has {len(wand_imgs.sequence)} pages")
        for wand_img in wand_imgs.sequence:
            wand_img.background_color = Color("white")
            wand_img.alpha_channel = "remove"
            numpy_array = np.array(wand_img)
            images.append(PILImage.fromarray(numpy_array).convert("RGB"))
    return images
