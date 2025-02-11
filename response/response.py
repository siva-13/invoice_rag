from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List, Optional
from services import user_service, pdf_service, invoice_service
from database import get_db
from models import User
from main import access_token, QueryRequest

router=APIRouter()

# User endpoints
@router.post("/signup", response_model=access_token)
async def signup(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    return user_service.create_user(db, form_data)

@router.post("/token", response_model=access_token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    return user_service.authenticate_user(db, form_data)

@router.get("/users/me")
async def get_user_info(current_user: User = Depends(user_service.get_current_user)):
    return user_service.get_user_info(current_user)

# PDF endpoints
@router.post("/upload-pdfs")
async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return pdf_service.upload_pdfs(db, current_user, files)

@router.get("/my-pdfs")
async def get_user_pdfs(
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return pdf_service.get_user_pdfs(db, current_user)

@router.get("/convert-pdfs-to-images-gpu")
async def convert_pdfs_to_images_gpu(
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return pdf_service.convert_pdfs_to_images_gpu(db, current_user)

# Invoice endpoints
@router.post("/process-invoices")
async def process_invoices(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_service.process_invoices(background_tasks, current_user, db)

@router.get("/processing-status/{processing_id}")
async def get_processing_status(
    processing_id: int,
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_service.get_processing_status(processing_id, current_user, db)

@router.get("/invoices")
async def get_user_invoices(
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
    sort_by: Optional[str] = "created_at",
    sort_order: Optional[str] = "desc"
):
    return invoice_service.get_user_invoices(current_user, db, skip, limit, sort_by, sort_order)

@router.get("/invoices/{invoice_id}")
async def get_invoice_detail(
    invoice_id: int,
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_service.get_invoice_detail(invoice_id, current_user, db)

@router.post("/query-invoices")
async def query_invoices(
    query: QueryRequest,
    current_user: User = Depends(user_service.get_current_user),
    db: Session = Depends(get_db)
):
    return invoice_service.query_invoices(query, current_user, db)