import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial 
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Dict
from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key
from paddleocr import PaddleOCR
import fitz


ocr_model=PaddleOCR(use_angle_cls=True,lang="en")
def extract_text_from_images(image_paths):
    """Extract text from images using PaddleOCR."""
    extracted_texts=[]
    for img_path in image_paths:
        try:
            result=ocr_model.ocr(img_path,cls=True)
            text=" ".join(word_info[1][0] for line in result for word_info in line)
            extracted_texts.append(text.strip())
        except Exception as e:
            print(f"Error processing{img_path}:{e}")
            extracted_texts.append("")
            print(extracted_texts)
    return extracted_texts


class GPUPDFProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.device = DEVICE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
    
    async def convert_pdf_to_images(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
        try:
            convert_func = partial(self._convert_single_pdf, output_dir=output_dir, filename=filename)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(self.executor, convert_func, pdf_path)
            return result
        except Exception as e:
            print(f"Error converting PDF {filename}: {str(e)}")
            return []

    def process_image_batch_gpu(self, images: List[Image.Image]) -> List[Image.Image]:
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        batch = batch.cpu()
        return [transforms.ToPILImage()(img) for img in batch]

    # def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> list:
    #     """Convert PDF to images with GPU acceleration without altering appearance."""
    #     try:
    #         # Ensure correct path format
    #         pdf_path = pdf_path.replace("\\", "/")

    #         # Check if the file exists
    #         if not os.path.exists(pdf_path):
    #             print(f"Error in _convert_single_pdf: File not found - {pdf_path}")
    #             return []

    #         # Open PDF
    #         doc = fitz.open(pdf_path)
    #         saved_paths = []
    #         base_filename = os.path.splitext(filename)[0]

    #         # Process images in batches using GPU
    #         for i in range(0, len(doc), BATCH_SIZE):
    #             batch = doc[i:i + BATCH_SIZE]
    #             processed_batch = []

    #             for j, page in enumerate(batch):
    #                 # Convert page to an image
    #                 pix = page.get_pixmap()
    #                 img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #                 processed_batch.append(img)

    #             # Pass the actual images, not raw pages
    #             processed_batch = self.process_image_batch_gpu(processed_batch)

    #             # Save processed images
    #             for j, processed_img in enumerate(processed_batch):
    #                 page_num = i + j + 1
    #                 image_path = os.path.join(output_dir, f"{base_filename}_page_{page_num}.jpg")
    #                 processed_img.save(image_path, 'JPEG', quality=90, optimize=True)
    #                 saved_paths.append(image_path)

    #         return saved_paths

    #     except Exception as e:
    #         print(f"Error in _convert_single_pdf: {str(e)}")
    #         return []

    def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> list:
        """Convert PDF to images with GPU acceleration without altering appearance and extract text via OCR."""
        try:
            # Ensure correct path format
            pdf_path = pdf_path.replace("\\", "/")

            # Check if the file exists
            if not os.path.exists(pdf_path):
                print(f"Error in _convert_single_pdf: File not found - {pdf_path}")
                return []

            # Open PDF
            doc = fitz.open(pdf_path)
            saved_paths = []
            base_filename = os.path.splitext(filename)[0]

            # Process images in batches using GPU
            for i in range(0, len(doc), BATCH_SIZE):
                batch = doc[i:i + BATCH_SIZE]
                processed_batch = []

                for j, page in enumerate(batch):
                    # Convert page to an image
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    processed_batch.append(img)

                # Process images through GPU
                processed_batch = self.process_image_batch_gpu(processed_batch)

                # Save processed images & Run OCR
                image_texts = []  # ✅ Store OCR results
                for j, processed_img in enumerate(processed_batch):
                    page_num = i + j + 1
                    image_path = os.path.join(output_dir, f"{base_filename}_page_{page_num}.jpg")
                    processed_img.save(image_path, 'JPEG', quality=90, optimize=True)
                    saved_paths.append(image_path)

                    # ✅ Extract text from the processed image
                    text = extract_text_from_images([image_path])[0]  # Only extract one image
                    image_texts.append({"page": page_num, "text": text})

                # ✅ Log OCR text for debugging
                print(f"OCR Extracted Text for {filename}: {image_texts}")

            return saved_paths

        except Exception as e:
            print(f"Error in _convert_single_pdf: {str(e)}")
            return []



class GPUPDFConversionManager:
    def __init__(self):
        self.processor = GPUPDFProcessor()
    
    # async def process_pdf_batch(self, pdf_files: List[dict], user_id: str) -> List[dict]:
    #     user_image_dir = os.path.join("./pdfs_to_image", user_id)
    #     os.makedirs(user_image_dir, exist_ok=True)
    #     results = []
    #     for i in range(0, len(pdf_files), 10):
    #         batch = pdf_files[i:i+10]
    #         tasks = [self.processor.convert_pdf_to_images(pdf['file_path'], user_image_dir, pdf['filename']) for pdf in batch]
    #         batch_results = await asyncio.gather(*tasks)
    #         for pdf, image_paths in zip(batch, batch_results):
    #             results.append({
    #                 'pdf_name': pdf['filename'],
    #                 'image_paths': image_paths,
    #                 'status': 'success' if image_paths else 'failed',
    #                 'pages_converted': len(image_paths)
    #             })
    #     return results

    async def process_pdf_batch(self, pdf_files: List[dict], user_id: str) -> List[dict]:
        user_image_dir = os.path.join("./pdfs_to_image", user_id)
        os.makedirs(user_image_dir, exist_ok=True)
        results = []
        successful_count = 0

        for i in range(0, len(pdf_files), 10):
            batch = pdf_files[i:i + 10]
            tasks = [self.processor.convert_pdf_to_images(pdf['file_path'], user_image_dir, pdf['filename']) for pdf in batch]
            batch_results = await asyncio.gather(*tasks)

            for pdf, image_paths in zip(batch, batch_results):
                if image_paths:
                    successful_count += len(image_paths)  # ✅ Count processed images

                results.append({
                    'pdf_name': pdf['filename'],
                    'image_paths': image_paths,
                    'status': 'success' if image_paths else 'failed',
                    'pages_converted': len(image_paths)
                })

        print(f"✅ Processed {successful_count} images successfully.")  # ✅ Debugging
        return results


gpu_conversion_manager = GPUPDFConversionManager()