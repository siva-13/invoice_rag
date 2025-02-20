import asyncio
from sqlalchemy.orm import Session
from database.database import SessionLocal
from services.openai_services import process_single_image
from database.models import ProcessingStatus, PDFFile
from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key
from datetime import datetime
import os

async def process_invoices_background(user_id: str, image_paths: list, pdf_file_ids: list, processing_status_id: int):
    db = SessionLocal()
    try:
        for i in range(0, len(image_paths), 5):
            batch = image_paths[i:i+5]
            pdf_files = db.query(PDFFile).filter(PDFFile.id.in_(pdf_file_ids)).all()
            tasks = []
            for pdf_file in pdf_files:
                base_filename = os.path.splitext(pdf_file.filename)[0]
                pdf_images = [path for path in batch if os.path.basename(path).startswith(base_filename)]
                for image_path in pdf_images:
                    tasks.append(process_single_image(image_path, db, pdf_file, processing_status_id))
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)
        status = db.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.status = 'completed'
            status.end_time = datetime.utcnow()
            db.commit()
    except Exception as e:
        status = db.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.status = 'failed'
            status.error_message = str(e)
            db.commit()
    finally:
        db.close()

def format_processing_job(status: ProcessingStatus):
    return {
        "processing_id": status.id,
        "status": status.status,
        "total_images": status.total_images,
        "processed_images": status.processed_images,
        "failed_images": status.failed_images,
        "progress_percentage": round((status.processed_images + status.failed_images) / status.total_images * 100, 2) if status.total_images > 0 else 0,
        "start_time": status.start_time,
        "end_time": status.end_time,
        "error_message": status.error_message,
        "duration": str(status.end_time - status.start_time) if status.end_time else None,
        "remaining_images": status.total_images - (status.processed_images + status.failed_images)
    }