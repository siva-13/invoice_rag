from sqlalchemy.orm import Session
from database.models import ProcessingStatus,User
from typing import Optional
from fastapi import HTTPException,status


# Add new utility function to format processing job info
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




def get_all_processing_jobs(
    db: Session,
    current_user: User,
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None
):
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

def get_active_processing_jobs(db: Session, current_user: User):
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

def get_processing_jobs_summary(db: Session, current_user: User):
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