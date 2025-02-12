import asyncio
from datetime import datetime
from typing import Optional
from scipy import stats
from sqlalchemy import text
from sqlalchemy.orm import Session
from database.database import SessionLocal, get_db
from fastapi import BackgroundTasks, Depends, HTTPException,status
from main import PDF_IMAGE_DIR, QueryRequest, generate_sql_query,client, synthesize_response
from database.models import PDFFile, User
from database.models import InvoiceDB, InvoiceItemDB, ProcessingStatus
from services.image_services import process_single_image, format_processing_job
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from services.auth import get_current_user


async def process_invoices_background(
    user_id: str,
    image_paths: list,
    pdf_file_ids: list,
    processing_status_id: int
):
    """Background task to process invoices with improved concurrency"""
    db = SessionLocal()
    try:
        # Process images in smaller batches to maintain responsiveness
        batch_size = 5
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            
            # Get PDF files for this batch
            pdf_files = db.query(PDFFile).filter(PDFFile.id.in_(pdf_file_ids)).all()
            
            tasks = []
            for pdf_file in pdf_files:
                base_filename = os.path.splitext(pdf_file.filename)[0]
                pdf_images = [
                    path for path in batch 
                    if os.path.basename(path).startswith(base_filename)
                ]
                
                for image_path in pdf_images:
                    task = process_single_image(
                        image_path,
                        db,
                        pdf_file,
                        processing_status_id
                    )
                    tasks.append(task)

            # Process batch with gather
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Small delay between batches to prevent overload
            await asyncio.sleep(0.1)

        # Update final status
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
            status.end_time = datetime.utcnow()
            db.commit()
    finally:
        db.close()

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

        # Get all relevant image paths
        image_paths = [
            os.path.join(user_image_dir, f) 
            for f in os.listdir(user_image_dir) 
            if f.endswith('.jpg')
        ]

        if not image_paths:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        # Create processing status record
        processing_status = ProcessingStatus(
            user_id=current_user.unique_id,
            total_images=len(image_paths),
            status='processing'
        )
        db.add(processing_status)
        db.commit()
        db.refresh(processing_status)

        # Start background processing
        background_tasks.add_task(
            process_invoices_background,
            current_user.unique_id,
            image_paths,
            pdf_file_ids,
            processing_status.id
        )

        return {
            "status": "processing_started",
            "message": "Invoice processing started in background",
            "total_images": len(image_paths),
            "processing_id": processing_status.id
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting invoice processing: {str(e)}"
        )
 
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
            detail=f"Error retrieving invoice details: {str(e)}"
        )

async def query_invoices(
    query: QueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process natural language queries about invoices and return results
    Example queries:
    - "What's my total invoice amount for last month?"
    - "Show me all invoices from seller X"
    - "What's the average invoice amount?"
    """
    try:
        print(query)
        # Generate SQL query from natural language
        sql_query = await generate_sql_query(query, client, current_user.unique_id)
        
        # Add user_id filter if not present (security measure)
        if "WHERE" not in sql_query.upper():
            sql_query = f"{sql_query} WHERE user_id = {current_user.unique_id}"
        elif "user_id" not in sql_query:
            sql_query = sql_query.replace("WHERE", f"WHERE user_id = {current_user.unique_id} AND")
            
        # Execute query with parameters
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: db.execute(
                text(sql_query),
                {"user_id": current_user.unique_id}
            ).fetchall()
        )
        print(f"sql response: {result}")
        # Convert result to list of dictionaries
        results = [
            {column: value for column, value in zip(row._mapping.keys(), row._mapping.values())}
            for row in result
        ]
        
        # Synthesize natural language response
        explanation = await synthesize_response(query, results, client)
        
        return {
            "query": query,
            "sql_query": sql_query,
            "results": results,
            "explanation": explanation
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )
