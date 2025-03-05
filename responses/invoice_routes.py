import asyncio
from fastapi import Depends,HTTPException,status,APIRouter
from typing import Any, Dict, List, Optional
from openai import OpenAI
from sqlalchemy.orm import Session
from database.models import InvoiceDB, InvoiceItemDB, PDFFile, QueryRequest, User,Query
from database.database import get_db
from services.auth import get_current_user
from sqlalchemy import text
from config import client
from services.invoice_services import generate_sql_query,synthesize_response



router=APIRouter()





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
            detail=f"Error retrieving invoice details: {str(e)}"
        )
    
async def generate_sql_query(query: str, client: OpenAI, user_id: str) -> str:
    """Convert natural language query to SQL using OpenAI"""

    schema = """
    Tables:
    - invoices: id, user_id, invoice_number, seller_name, seller_gstin, date_of_invoice, 
                buyer_order_number, buyer_name, buyer_gstin, number_of_items, 
                total_amount, sgst, cgst, created_at
    - invoice_items: id, invoice_id, description, quantity, rate_per_unit, amount
    
    Relationships:
    - invoices has many invoice_items (one-to-many)
    - Both tables are filtered by user_id for security
    """

    response = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a SQL expert. Convert natural language queries to SQL based on this schema:\n{schema}\nOnly return the SQL query, nothing else."
            },
            {
                "role": "user",
                "content": f"Convert this question to SQL (always include user_id filter: {user_id}): {query}"
            }
        ],
        response_format=Query,
    )
    )
    
    invoice_data = response.choices[0].message.parsed
    print(invoice_data.sqlQuery)
    return invoice_data.sqlQuery

async def synthesize_response(user_question: str, sql_query_results: List[Dict[str, Any]], client: OpenAI) -> str:
    """Convert SQL results to natural language response using OpenAI"""
    response = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Please generate a clear and concise human-readable response based on the user's question and the SQL query results. If no results are found, explain this to the user."},
                {"role": "user", "content": f"User's Question: {user_question} SQL Query Results: {sql_query_results}."}
            ],
            temperature=0.1
        )
    )
    
    return response.choices[0].message.content.strip()

@router.post("/query-invoices")
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

