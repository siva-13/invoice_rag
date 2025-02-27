import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
# from pdf2image import convert_from_path
import fitz
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from database.models import User, PDFFile
from fastapi import Depends,HTTPException,status,UploadFile,File
from database.database import get_db,engine,Base
from datetime import datetime, time
import shutil
from typing import List,Optional
import os
import torch
from PIL import Image
from torchvision import transforms
from config import DEVICE,UPLOAD_DIR,PDF_IMAGE_DIR,MAX_WORKERS,BATCH_SIZE,API_SEMAPHORE,client
from text import extract_text_from_images
from database.models import PDFFile,ProcessingStatus,Invoice,InvoiceDB,InvoiceItemDB


# Create directory for uploaded PDFs
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class GPUPDFProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.device = DEVICE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
       
    async def convert_pdf_to_images(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
        try:
            convert_func = partial(
                self._convert_single_pdf,
                output_dir=output_dir,
                filename=filename
            )
           
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                convert_func,
                pdf_path
            )
            return result
           
        except Exception as e:
            print(f"Error converting PDF {filename}: {str(e)}")
            return []

    def process_image_batch_gpu(self, images: List[Image.Image]) -> List[Image.Image]:
        """Process a batch of images using GPU, but retain their original appearance."""
        # Convert images to tensors and move to GPU without altering appearance
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        # No processing like contrast enhancement, just convert back to CPU and PIL Images
        batch = batch.cpu()
        processed_images = [
            transforms.ToPILImage()(img)
            for img in batch
        ]
       
        return processed_images

    # def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
    #     """Convert PDF to images with GPU acceleration without altering appearance."""
    #     try:
    #         # Convert PDF pages to images
    #         images = convert_from_path(
    #             pdf_path,
    #             dpi=200,
    #             fmt='jpeg',
    #             thread_count=2
    #         )
           
    #         saved_paths = []
    #         base_filename = os.path.splitext(filename)[0]
           
    #         # Process images in batches using GPU
    #         for i in range(0, len(images), BATCH_SIZE):
    #             batch = images[i:i + BATCH_SIZE]
    #             processed_batch = self.process_image_batch_gpu(batch)
               
    #             # Save processed images
    #             for j, processed_img in enumerate(processed_batch):
    #                 page_num = i + j + 1
    #                 image_path = os.path.join(
    #                     output_dir,
    #                     f"{base_filename}_page_{page_num}.jpg"
    #                 )
    #                 processed_img.save(
    #                     image_path,
    #                     'JPEG',
    #                     quality=90,
    #                     optimize=True
    #                 )
    #                 saved_paths.append(image_path)
           
    #         return saved_paths
           
    #     except Exception as e:
    #         print(f"Error in _convert_single_pdf: {str(e)}")
    #         return []

    def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> dict:
        """Convert PDF to images and extract text."""
        try:
            pdf_path = pdf_path.replace("\\", "/")
            if not os.path.exists(pdf_path):
                print(f"File not found: {pdf_path}")
                return {'image_paths': [], 'extracted_texts': []}

            doc = fitz.open(pdf_path)
            saved_paths = []
            extracted_texts = []
            base_filename = os.path.splitext(filename)[0]

            for i in range(0, len(doc), BATCH_SIZE):
                batch = doc[i:i + BATCH_SIZE]
                processed_batch = []

                for page in batch:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    processed_batch.append(img)

                processed_batch = self.process_image_batch_gpu(processed_batch)

                for j, processed_img in enumerate(processed_batch):
                    page_num = i + j + 1
                    image_path = os.path.join(output_dir, f"{base_filename}_page_{page_num}.jpg")
                    processed_img.save(image_path, 'JPEG', quality=90, optimize=True)
                    saved_paths.append(image_path)

                # Extract text from this batch of images
                batch_texts = extract_text_from_images(saved_paths[-len(processed_batch):])
                extracted_texts.extend(batch_texts)

            return {'image_paths': saved_paths, 'extracted_texts': extracted_texts}
        except Exception as e:
            print(f"Error in _convert_single_pdf: {str(e)}")
            return {'image_paths': [], 'extracted_texts': []}


class GPUPDFConversionManager:
    def __init__(self):
        self.processor = GPUPDFProcessor()
       
    async def process_pdf_batch(
        self,
        pdf_files: List[dict],
        user_id: str
    ) -> List[dict]:
        # Create user directory
        user_image_dir = os.path.join(PDF_IMAGE_DIR, user_id)
        os.makedirs(user_image_dir, exist_ok=True)
       
        # Process PDFs in optimized batches
        results = []
        for i in range(0, len(pdf_files), BATCH_SIZE):
            batch = pdf_files[i:i + BATCH_SIZE]
            tasks = [
                self.processor.convert_pdf_to_images(
                    pdf['file_path'],
                    user_image_dir,
                    pdf['filename']
                )
                for pdf in batch
            ]
           
            batch_results = await asyncio.gather(*tasks)
           
            for pdf, image_paths in zip(batch, batch_results):
                results.append({
                    'pdf_name': pdf['filename'],
                    'image_paths': image_paths,
                    'status': 'success' if image_paths else 'failed',
                    'pages_converted': len(image_paths)
                })
               
        return results

# Initialize GPU-enabled manager
gpu_conversion_manager = GPUPDFConversionManager()
    

