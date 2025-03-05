import asyncio
from datetime import datetime
from typing import Optional,List,Dict,Any
from groq import Groq
from openai import OpenAI
from scipy import stats
from sqlalchemy import text
from sqlalchemy.orm import Session
from database.database import SessionLocal, get_db
from fastapi import BackgroundTasks, Depends, HTTPException, Query,status
from database.models import PDFFile, User
from database.models import InvoiceDB, InvoiceItemDB, ProcessingStatus,QueryRequest
#from services.pdf_services import process_single_image
from services.processing_services import format_processing_job
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from services.auth import get_current_user
from config import PDF_IMAGE_DIR,client
from pydantic import BaseModel

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




