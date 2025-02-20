from fastapi import APIRouter, Depends, HTTPException, File,status
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import InvoiceDB, InvoiceItemDB, PDFFile,User
from services.auth import get_current_user
from typing import List,Optional
from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key


router = APIRouter(tags=["Invoices"])


@router.get("/invoices")
async def get_user_invoices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
    sort_by: Optional[str] = "created_at",
    sort_order: Optional[str] = "desc"
):
    """
    Retrieve all invoices for the current user with pagination and sorting options
    """
    try:
        # Base query
        query = db.query(InvoiceDB).filter(
            InvoiceDB.user_id == current_user.unique_id
        )
        
        # Apply sorting
        sort_field = getattr(InvoiceDB, sort_by, InvoiceDB.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(sort_field.desc())
        else:
            query = query.order_by(sort_field.asc())
            
        # Get total count for pagination
        total_invoices = query.count()
        
        # Apply pagination
        invoices = query.offset(skip).limit(limit).all()
        
        # Format response with detailed invoice information
        formatted_invoices = []
        for invoice in invoices:
            # Get all items for this invoice
            items = db.query(InvoiceItemDB).filter(
                InvoiceItemDB.invoice_id == invoice.id
            ).all()
            
            formatted_items = [
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "rate_per_unit": item.rate_per_unit,
                    "amount": item.amount
                }
                for item in items
            ]
            
            # Get associated PDF file information
            pdf_file = db.query(PDFFile).filter(
                PDFFile.id == invoice.pdf_file_id
            ).first()
            
            formatted_invoices.append({
                "id": invoice.id,
                "invoice_number": invoice.invoice_number,
                "seller_name": invoice.seller_name,
                "seller_gstin": invoice.seller_gstin,
                "date_of_invoice": invoice.date_of_invoice,
                "buyer_order_number": invoice.buyer_order_number,
                "buyer_name": invoice.buyer_name,
                "buyer_gstin": invoice.buyer_gstin,
                "number_of_items": invoice.number_of_items,
                "total_amount": invoice.total_amount,
                "sgst": invoice.sgst,
                "cgst": invoice.cgst,
                "created_at": invoice.created_at,
                "items": formatted_items,
                "pdf_info": {
                    "filename": pdf_file.filename if pdf_file else None,
                    "upload_time": pdf_file.upload_time if pdf_file else None
                }
            })
        
        # Calculate summary statistics
        total_amount = sum(invoice.total_amount or 0 for invoice in invoices)
        total_items = sum(invoice.number_of_items or 0 for invoice in invoices)
        
        return {
            "invoices": formatted_invoices,
            "pagination": {
                "total": total_invoices,
                "skip": skip,
                "limit": limit,
                "has_more": (skip + limit) < total_invoices
            },
            "summary": {
                "total_invoices": total_invoices,
                "total_amount": total_amount,
                "total_items": total_items,
                "average_amount": total_amount / len(invoices) if invoices else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving invoices: {str(e)}"
        )

@router.get("/invoices/{invoice_id}")
async def get_invoice_detail(
    invoice_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve detailed information for a specific invoice
    """
    try:
        # Get invoice with verification of ownership
        invoice = db.query(InvoiceDB).filter(
            InvoiceDB.id == invoice_id,
            InvoiceDB.user_id == current_user.unique_id
        ).first()
        
        if not invoice:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found"
            )
        
        # Get invoice items
        items = db.query(InvoiceItemDB).filter(
            InvoiceItemDB.invoice_id == invoice.id
        ).all()
        
        # Get associated PDF file
        pdf_file = db.query(PDFFile).filter(
            PDFFile.id == invoice.pdf_file_id
        ).first()
        
        return {
            "invoice_details": {
                "id": invoice.id,
                "invoice_number": invoice.invoice_number,
                "seller_name": invoice.seller_name,
                "seller_gstin": invoice.seller_gstin,
                "date_of_invoice": invoice.date_of_invoice,
                "buyer_order_number": invoice.buyer_order_number,
                "buyer_name": invoice.buyer_name,
                "buyer_gstin": invoice.buyer_gstin,
                "number_of_items": invoice.number_of_items,
                "total_amount": invoice.total_amount,
                "sgst": invoice.sgst,
                "cgst": invoice.cgst,
                "created_at": invoice.created_at
            },
            "items": [
                {
                    "description": item.description,
                    "quantity": item.quantity,
                    "rate_per_unit": item.rate_per_unit,
                    "amount": item.amount
                }
                for item in items
            ],
            "pdf_info": {
                "filename": pdf_file.filename if pdf_file else None,
                "upload_time": pdf_file.upload_time if pdf_file else None
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving invoice details: {str(e)}")


@router.get("/my-pdfs")
async def get_user_pdfs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        pdf_files = db.query(PDFFile).filter(
            PDFFile.user_id == current_user.unique_id
        ).all()
        
        return {
            "status": "success",
            "user_id": current_user.unique_id,
            "files": [
                {
                    "filename": pdf.filename,
                    "upload_time": pdf.upload_time
                } for pdf in pdf_files
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving PDF files: {str(e)}"
        )