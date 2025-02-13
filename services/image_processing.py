import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from pdf2image import convert_from_path
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Dict

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

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

    def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
        try:
            images = convert_from_path(pdf_path, dpi=200, fmt='jpeg', thread_count=2)
            saved_paths = []
            base_filename = os.path.splitext(filename)[0]
            for i in range(0, len(images), 10):
                batch = images[i:i+10]
                processed_batch = self.process_image_batch_gpu(batch)
                for j, img in enumerate(processed_batch):
                    page_num = i + j + 1
                    image_path = os.path.join(output_dir, f"{base_filename}_page_{page_num}.jpg")
                    img.save(image_path, 'JPEG', quality=90, optimize=True)
                    saved_paths.append(image_path)
            return saved_paths
        except Exception as e:
            print(f"Error in _convert_single_pdf: {str(e)}")
            return []

class GPUPDFConversionManager:
    def __init__(self):
        self.processor = GPUPDFProcessor()
    
    async def process_pdf_batch(self, pdf_files: List[dict], user_id: str) -> List[dict]:
        user_image_dir = os.path.join("./pdfs_to_image", user_id)
        os.makedirs(user_image_dir, exist_ok=True)
        results = []
        for i in range(0, len(pdf_files), 10):
            batch = pdf_files[i:i+10]
            tasks = [self.processor.convert_pdf_to_images(pdf['file_path'], user_image_dir, pdf['filename']) for pdf in batch]
            batch_results = await asyncio.gather(*tasks)
            for pdf, image_paths in zip(batch, batch_results):
                results.append({
                    'pdf_name': pdf['filename'],
                    'image_paths': image_paths,
                    'status': 'success' if image_paths else 'failed',
                    'pages_converted': len(image_paths)
                })
        return results

gpu_conversion_manager = GPUPDFConversionManager()