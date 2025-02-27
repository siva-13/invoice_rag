from paddleocr import PaddleOCR
from typing import List

import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)  # Suppress DEBUG logs
logging.getLogger("paddle").setLevel(logging.ERROR)  # Suppress PaddleOCR logs



ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

def extract_text_from_images(image_paths: List[str]) -> List[str]:
    """Extract text from images using PaddleOCR."""
    extracted_texts = []
    for img_path in image_paths:
            try:
                result = ocr_model.ocr(img_path, cls=True)
                if result and isinstance(result, list):
                    text = " ".join(word_info[1][0] for line in result for word_info in line if line)
                    extracted_texts.append(text.strip())
                else:
                    extracted_texts.append("")
                print(f"Extracted Text from {img_path}: {extracted_texts[-1]}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                extracted_texts.append("")
    return extracted_texts
        