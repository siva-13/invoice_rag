import os
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, File, UploadFile, status
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import ProcessingStatus, PDFFile,User
from services.auth import get_current_user
from services.background_tasks import process_invoices_background, format_processing_job
from services.image_processing import gpu_conversion_manager
import shutil
import time
from typing import Optional, List, Any, Dict
from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key
from services.image_processing import extract_text_from_images


router = APIRouter(tags=["Processing"])

@router.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create user directory if it doesn't exist
    user_dir = os.path.join(UPLOAD_DIR, current_user.unique_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    uploaded_files = []
    
    try:
        for file in files:
            # Verify if file is PDF
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a PDF"
                )
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(user_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Save file info to database
            pdf_file = PDFFile(
                filename=unique_filename,
                file_path=file_path,
                user_id=current_user.unique_id
            )
            db.add(pdf_file)
            
            uploaded_files.append({
                "original_filename": file.filename,
                "saved_filename": unique_filename
            })
        
        db.commit()
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "user_id": current_user.unique_id,
            "files": uploaded_files
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}"
        )

@router.get("/convert-pdfs-to-images-gpu")
async def convert_pdfs_to_images_gpu(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get user's PDFs
        pdf_files = db.query(PDFFile).filter(
            PDFFile.user_id == current_user.unique_id
        ).all()
       
        if not pdf_files:
            return {
                "status": "error",
                "message": "No PDFs found for conversion"
            }
        
        # ✅ Ensure all PDF files exist before processing
        for pdf in pdf_files:
            if not os.path.exists(pdf.file_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"File {pdf.file_path} not found"
                )

        # Process PDFs with GPU acceleration (✅ Added `await`)
        start_time = time.time()
        conversion_results = await gpu_conversion_manager.process_pdf_batch(  # ✅ Await the coroutine
            [{"file_path": pdf.file_path, "filename": pdf.filename} for pdf in pdf_files],
            current_user.unique_id
        )
        end_time = time.time()

        successful_conversions = [
            result for result in conversion_results if result['status'] == 'success'
        ]
        total_pages = sum(result['pages_converted'] for result in successful_conversions)

        return {
            "status": "success",
            "message": f"Converted {len(successful_conversions)} out of {len(pdf_files)} PDFs",
            "total_pages_converted": total_pages,
            "processing_time": f"{end_time - start_time:.2f} seconds",
            "processing_device": str(DEVICE),
            "results": conversion_results
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during GPU-accelerated PDF conversion: {str(e)}"
        )


@router.get("/processing-status/{processing_id}")
async def get_processing_status(
    processing_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of a processing job"""
    status = db.query(ProcessingStatus).filter(
        ProcessingStatus.id == processing_id,
        ProcessingStatus.user_id == current_user.unique_id
    ).first()

    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing job not found"
        )

    return {
        "status": status.status,
        "total_images": status.total_images,
        "processed_images": status.processed_images,
        "failed_images": status.failed_images,
        "start_time": status.start_time.isoformat() if status.start_time else None,  # ✅ Fix for None timestamps
        "end_time": status.end_time.isoformat() if status.end_time else None,
        "error_message": status.error_message
    }

# @router.post("/process-invoices")
# async def process_invoices(
#     background_tasks: BackgroundTasks,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
#         if not os.path.exists(user_image_dir) or not os.listdir(user_image_dir):  # ✅ Fix: Ensure folder exists
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No images found for processing"
#             )

#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()

#         if not pdf_files:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No PDF files found"
#             )

#         pdf_file_ids = [pdf.id for pdf in pdf_files]
#         image_paths = [os.path.join(user_image_dir, f) for f in os.listdir(user_image_dir) if f.endswith('.jpg')]

#         if not image_paths:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No images found for processing"
#             )

#         processing_status = ProcessingStatus(
#             user_id=current_user.unique_id,
#             total_images=len(image_paths),
#             status='processing'
#         )
#         db.add(processing_status)
#         db.commit()
#         db.refresh(processing_status)

#         background_tasks.add_task(
#             process_invoices_background,
#             current_user.unique_id,
#             image_paths,
#             pdf_file_ids,
#             processing_status.id
#         )

#         return {
#             "status": "processing_started",
#             "message": "Invoice processing started in background",
#             "total_images": len(image_paths),
#             "processing_id": processing_status.id
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error starting invoice processing: {str(e)}"
#         )



from fastapi import BackgroundTasks

async def process_ocr_background(image_paths: List[str], processing_status_id: int, db: Session):
    """Process OCR in the background and update the processing status."""
    try:
        extracted_texts = extract_text_from_images(image_paths)
        
        # Update the processing status in the database
        processing_status = db.query(ProcessingStatus).filter(
            ProcessingStatus.id == processing_status_id
        ).first()
        
        if processing_status:
            processing_status.processed_images = len(extracted_texts)
            processing_status.status = "completed"
            db.commit()
        
        # Save extracted text to the database or perform other actions
        # Example: Save to a TextFile model or return as part of the response
        
    except Exception as e:
        # Handle errors and update the status
        processing_status = db.query(ProcessingStatus).filter(
            ProcessingStatus.id == processing_status_id
        ).first()
        
        if processing_status:
            processing_status.status = "failed"
            processing_status.error_message = str(e)
            db.commit()
        raise

@router.post("/process-invoices")
async def process_invoices(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
        if not os.path.exists(user_image_dir) or not os.listdir(user_image_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        image_paths = [os.path.join(user_image_dir, f) for f in os.listdir(user_image_dir) if f.endswith('.jpg')]

        if not image_paths:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        # Create a new processing status record
        processing_status = ProcessingStatus(
            user_id=current_user.unique_id,
            total_images=len(image_paths),
            status='processing'
        )
        db.add(processing_status)
        db.commit()
        db.refresh(processing_status)

        # Add OCR processing to background tasks
        background_tasks.add_task(
            process_ocr_background,
            image_paths,
            processing_status.id,
            db
        )

        return {
            "status": "processing_started",
            "message": "OCR processing started in background",
            "total_images": len(image_paths),
            "processing_id": processing_status.id
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting OCR processing: {str(e)}"
        )


@router.get("/processing-jobs")
async def get_all_processing_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None
):
    """
    Get all processing jobs for the current user with optional filtering and pagination
    """
    try:
        # Base query
        query = db.query(ProcessingStatus).filter(
            ProcessingStatus.user_id == current_user.unique_id
        )
        
        # Apply status filter if provided
        if status_filter:
            query = query.filter(ProcessingStatus.status == status_filter)
        
        # Get total count for pagination
        total_jobs = query.count()
        
        # Get jobs with pagination and ordering
        jobs = query.order_by(ProcessingStatus.start_time.desc())\
                   .offset(skip)\
                   .limit(limit)\
                   .all()

        # Format response with summary statistics
        active_jobs = sum(1 for job in jobs if job.status == 'processing')
        completed_jobs = sum(1 for job in jobs if job.status == 'completed')
        failed_jobs = sum(1 for job in jobs if job.status == 'failed')
        
        total_images_processed = sum(job.processed_images for job in jobs)
        total_images_failed = sum(job.failed_images for job in jobs)
        
        return {
            "jobs": [format_processing_job(job) for job in jobs],
            "pagination": {
                "total": total_jobs,
                "skip": skip,
                "limit": limit,
                "has_more": (skip + limit) < total_jobs
            },
            "summary": {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "completed_jobs": completed_jobs,
                "failed_jobs": failed_jobs,
                "total_images_processed": total_images_processed,
                "total_images_failed": total_images_failed
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving processing jobs: {str(e)}"
        )

@router.get("/processing-jobs/active")
async def get_active_processing_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get only active processing jobs for the current user
    """
    try:
        active_jobs = db.query(ProcessingStatus).filter(
            ProcessingStatus.user_id == current_user.unique_id,
            ProcessingStatus.status == 'processing'
        ).order_by(ProcessingStatus.start_time.desc()).all()

        return {
            "active_jobs": [format_processing_job(job) for job in active_jobs],
            "count": len(active_jobs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving active jobs: {str(e)}"
        )

@router.get("/processing-jobs/summary")
async def get_processing_jobs_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a summary of all processing jobs for the current user
    Returns both summary statistics and a list of all formatted jobs
    """
    try:
        # Get all jobs for the user
        jobs = db.query(ProcessingStatus).filter(
            ProcessingStatus.user_id == current_user.unique_id
        ).all()
        
        # Calculate statistics
        total_jobs = len(jobs)
        status_counts = {
            'processing': 0,
            'completed': 0,
            'failed': 0
        }
        total_images = 0
        total_processed = 0
        total_failed = 0
        
        for job in jobs:
            status_counts[job.status] += 1
            total_images += job.total_images
            total_processed += job.processed_images
            total_failed += job.failed_images
        
        # Calculate success rate
        success_rate = (total_processed / total_images * 100) if total_images > 0 else 0
        
        # Format all jobs
        formatted_jobs = [format_processing_job(job) for job in jobs]
        
        return {
            "total_jobs": total_jobs,
            "status_breakdown": status_counts,
            "image_statistics": {
                "total_images": total_images,
                "processed_images": total_processed,
                "failed_images": total_failed,
                "success_rate": round(success_rate, 2)
            },
            "jobs": formatted_jobs 
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving jobs summary: {str(e)}"
        )