from paddleocr import PaddleOCR
from typing import List, Dict, Tuple
import logging
import os
import shutil
from pathlib import Path

# Suppress DEBUG logs
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)

# Initialize OCR model only once
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

def extract_text_from_images(image_paths: List[str]) -> Dict[str, str]:
    """
    Extract text from images using PaddleOCR and move processed images.
    
    Args:
        image_paths: List of paths to images to process
        
    Returns:
        Dictionary mapping image paths to extracted text
    """
    results = {}
    
    for img_path in image_paths:
        try:
            # Extract text from image
            result = ocr_model.ocr(img_path, cls=True)
            
            # Process OCR results
            if result and isinstance(result, list):
                text = " ".join(word_info[1][0] for line in result for word_info in line if line)
                extracted_text = text.strip()
            else:
                extracted_text = ""
                
            results[img_path] = extracted_text
            print(f"Extracted Text from {img_path}: {extracted_text}")
            
            # Move file to processed folder
            # move_to_processed_folder(img_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            results[img_path] = ""
    
    return results
