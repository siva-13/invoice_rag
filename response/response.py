from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import List, Optional
from services import pdf_services, user_services, invoice_services, processing_job_services
from database import get_db
from database.models import User
from services.auth import get_current_user

router = APIRouter()

# User endpoints
@router.post("/signup")
async def signup(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    return user_services.create_user(db, form_data)

@router.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    return user_services.authenticate_user(db, form_data)

@router.get("/users/me")
async def user_info(current_user: User = Depends(get_current_user)):
    return user_services.get_user_info(current_user)

# PDF endpoints
@router.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return pdf_services.upload_pdfs(db, current_user, files)

@router.get("/my-pdfs")
async def get_user_pdfs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return pdf_services.get_user_pdfs(db, current_user)

@router.get("/convert-pdfs-to-images-gpu")
async def convert_pdfs_to_images_gpu(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return pdf_services.convert_pdfs_to_images_gpu(db, current_user)

# Invoice endpoints
@router.post("/process-invoices")
async def process_invoices(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_services.process_invoices(background_tasks, current_user, db)

@router.get("/processing-status/{processing_id}")
async def get_processing_status(
    processing_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_services.get_processing_status(processing_id, current_user, db)

@router.get("/invoices")
async def get_user_invoices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
    sort_by: Optional[str] = "created_at",
    sort_order: Optional[str] = "desc"
):
    return invoice_services.get_user_invoices(current_user, db, skip, limit, sort_by, sort_order)

@router.get("/invoices/{invoice_id}")
async def get_invoice_detail(
    invoice_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_services.get_invoice_detail(invoice_id, current_user, db)

@router.post("/query-invoices")
async def query_invoices(
    query: dict,  # Removed `QueryRequest` schema
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_services.query_invoices(query, current_user, db)

# Processing job endpoints
@router.get("/processing-jobs")
async def get_all_processing_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    status_filter: Optional[str] = None
):
    return processing_job_services.get_all_processing_jobs(db, current_user, skip, limit, status_filter)

@router.get("/processing-jobs/active")
async def get_active_processing_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return processing_job_services.get_active_processing_jobs(db, current_user)

@router.get("/processing-jobs/summary")
async def get_processing_jobs_summary(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return processing_job_services.get_processing_jobs_summary(db, current_user)
