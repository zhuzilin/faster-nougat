from wand.image import Image as WandImage
import numpy as np
from PIL import Image as PILImage

def extract_pdf_as_image(filename: str, page_idx: int):
    assert filename.endswith(".pdf"), f"{filename} does not end with .pdf, is it a pdf file?"
    # Load the image using wand
    with WandImage(filename=f'{filename}[{page_idx}]', resolution=200) as wand_img:
        # Convert wand image to numpy array
        numpy_array = np.array(wand_img)
        # Create a new PIL Image object from the numpy array
        image = PILImage.fromarray(numpy_array).convert("RGB")
    return image
