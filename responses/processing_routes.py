import os
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi import BackgroundTasks, Depends,HTTPException,status,APIRouter
from database.database import SessionLocal, get_db
from config import API_SEMAPHORE, PDF_IMAGE_DIR,client
import asyncio
from typing import Optional
from database.models import Invoice, InvoiceDB, InvoiceItemDB, PDFFile, ProcessingStatus, User
from services.auth import get_current_user
from text import extract_text_from_images
from services.processing_services import process_invoices_background,process_single_image,format_processing_job


router=APIRouter()

@router.post("/process-invoices")   
async def process_invoices(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get all image paths for the user
        user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
        if not os.path.exists(user_image_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        # Get all PDF files for the user
        pdf_files = db.query(PDFFile).filter(
            PDFFile.user_id == current_user.unique_id
        ).all()

        if not pdf_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No PDF files found"
            )

        # Get PDF file IDs
        pdf_file_ids = [pdf.id for pdf in pdf_files]

        # Debugging: Verify InvoiceDB table query
        try:
            processed_pdf_ids = db.query(InvoiceDB.pdf_file_id).filter(
                InvoiceDB.pdf_file_id.in_(pdf_file_ids)
            ).distinct().all()
            processed_pdf_ids = {row[0] for row in processed_pdf_ids if row[0] is not None}  # Handle NULL values
        except Exception as query_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error querying invoices table: {str(query_error)}"
            )

        # Filter out PDFs that are already processed
        unprocessed_pdf_files = [
            pdf for pdf in pdf_files if pdf.id not in processed_pdf_ids
        ]

        if not unprocessed_pdf_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No new PDF files found for processing (all already processed)"
            )

        # Create mapping of PDF filename to ID
        pdf_filename_to_id = {pdf.filename: pdf.id for pdf in pdf_files}

        # Get all relevant image paths and filter for unprocessed PDFs only
        image_paths = [
            os.path.join(user_image_dir, f)
            for f in os.listdir(user_image_dir)
            if f.endswith('.jpg')
        ]

        filtered_image_paths = []
        for image_path in image_paths:
            # Extract base filename from image (e.g., "invoice1_page_1.jpg" -> "invoice1.pdf")
            filename = os.path.basename(image_path)
            base_filename = "_".join(filename.split("_")[:-2]) + ".pdf"  # Assuming PDF extension
            pdf_id = pdf_filename_to_id.get(base_filename)
            
            # Include only if PDF is unprocessed
            if pdf_id and pdf_id not in processed_pdf_ids:
                filtered_image_paths.append((image_path, pdf_id))

        if not filtered_image_paths:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No new images found for processing (all PDFs already processed)"
            )

        # Create processing status record
        processing_status = ProcessingStatus(
            user_id=current_user.unique_id,
            total_images=len(filtered_image_paths),
            status='processing'
        )
        db.add(processing_status)
        db.commit()
        db.refresh(processing_status)

        # Process each image individually
        for image_path, pdf_id in filtered_image_paths:
            # Extract text from single image
            extracted_text = extract_text_from_images([image_path])
            
            # Start background processing for this image
            background_tasks.add_task(
                process_invoices_background,
                current_user.unique_id,
                extracted_text,
                [pdf_id],  # Pass single PDF ID as a list
                processing_status.id
            )

        return {
            "status": "processing_started",
            "message": "Invoice processing started in background for each image individually",
            "total_images": len(filtered_image_paths),
            "processing_id": processing_status.id
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting invoice processing: {str(e)}"
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
        "start_time": status.start_time,
        "end_time": status.end_time,
        "error_message": status.error_message
    }



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
            "jobs": formatted_jobs  # Return all formatted jobs instead of just the latest
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving jobs summary: {str(e)}"
        )

