from concurrent.futures import ThreadPoolExecutor
from tkinter import Image
from main import BATCH_SIZE, MAX_WORKERS,DEVICE, PDF_IMAGE_DIR
import torch
from functools import partial
import asyncio
from typing import List
from pdf2image import convert_from_path
from torchvision import transforms
import os






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

    async def process_image_batch_gpu(self, images: List[Image.Image]) -> List[Image.Image]:
        """Process a batch of images using GPU, but retain their original appearance."""
        # Convert images to tensors and move to GPU without altering appearance
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        # No processing like contrast enhancement, just convert back to CPU and PIL Images
        batch = batch.cpu()
        processed_images =await [
            transforms.ToPILImage()(img)
            for img in batch
        ]
       
        return processed_images

    async def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
        """Convert PDF to images with GPU acceleration without altering appearance."""
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=200,
                fmt='jpeg',
                thread_count=2
            )
           
            saved_paths = []
            base_filename = os.path.splitext(filename)[0]
           
            # Process images in batches using GPU
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]
                processed_batch = self.process_image_batch_gpu(batch)
               
                # Save processed images
                for j, processed_img in enumerate(processed_batch):
                    page_num = i + j + 1
                    image_path = os.path.join(
                        output_dir,
                        f"{base_filename}_page_{page_num}.jpg"
                    )
                    processed_img.save(
                        image_path,
                        'JPEG',
                        quality=90,
                        optimize=True
                    )
                    await saved_paths.append(image_path)
           
            return saved_paths
           
        except Exception as e:
            print(f"Error in _convert_single_pdf: {str(e)}")
            return []

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
    
