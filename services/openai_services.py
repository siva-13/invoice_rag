import sys
import os
import base64
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sqlalchemy.orm import Session
from database.models import PDFFile, ProcessingStatus, InvoiceDB, InvoiceItemDB
from config import client, API_SEMAPHORE,llm
import asyncio
from services.image_processing import extract_text_from_images
from database.database import SessionLocal



class InvoiceStep(BaseModel):
    description: str = Field(..., description="Description of the item")
    quantity: int = Field(..., description="Quantity of the item")
    rate_per_unit: float = Field(..., description="Rate per unit of the item")
    amount: Optional[float] = Field(None, description="Total amount for the item")

class Invoice(BaseModel):
    invoice_number: str = Field(..., description="Unique invoice identifier")
    seller_name: str = Field(..., description="Name of the seller")
    seller_gstin: Optional[str] = Field(None, description="GSTIN of the seller")
    date_of_invoice: str = Field(..., description="Date of the invoice in YYYY-MM-DD format")
    buyer_order_number: Optional[str] = Field(None, description="Order number from the buyer")
    buyer_name: str = Field(..., description="Name of the buyer")
    buyer_gstin: Optional[str] = Field(None, description="GSTIN of the buyer")
    number_of_items: Optional[int] = Field(None, description="Number of items in the order")
    item_list: List[InvoiceStep]
    total_amount: Optional[float] = Field(None, description="Total amount for the invoice")
    sgst: Optional[float] = Field(None, description="State GST amount")
    cgst: Optional[float] = Field(None, description="Central GST amount")

class Query(BaseModel):
    sqlQuery: str = Field(..., description="SQL query based on user prompt")

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def process_single_image(extracted_text:str,pdf_file_id:int,user_id:int,processing_status_id:int)-> Optional[Invoice]:
    session=SessionLocal()
    try:
        async with API_SEMAPHORE:
            try:
                invoice_prompt = f"""
                Extract structured data from this invoice image. 
                Be precise and accurate in extracting the data. Check the image and extract:
                Invoice Number, Seller Name, Seller GSTIN, Date of Invoice, Buyer Order Number, Buyer Name, Buyer GSTIN, 
                Number of Items, Total Amount, SGST, CGST, and a list of items with Description, Quantity, Rate per Unit, and Amount.

                Invoice Text: {extracted_text}
                """
                response = await asyncio.wait_for(llm.invoke(invoice_prompt),timeout=25)
                # Validate the LLM response
                if not response or not hasattr(response, "content"):
                    print("Invalid LLM response.")
                    raise ValueError("Invalid LLM response.")
                invoice_data = response.choices[0].message.parsed
                required_fields = [
                    "invoice_number", "seller_name", "seller_gstin", "date_of_invoice",
                    "buyer_order_number", "buyer_name", "buyer_gstin", "number_of_items",
                    "total_amount", "sgst", "cgst", "item_list"
                ]
                for field in required_fields:
                    if field not in invoice_data:
                        print(f"Missing required field in LLM response: {field}")
                        raise ValueError(f"Missing required field: {field}")
                invoice=Invoice(**invoice_data)
                invoice_db=InvoiceDB(
                    user_id=user_id,
                    pdf_file_id=pdf_file_id,
                    invoice_number=invoice.invoice_number,
                    seller_name=invoice.seller_name,
                    seller_gstin=invoice.seller_gstin,
                    date_of_invoice=invoice.date_of_invoice,
                    buyer_order_number=invoice.buyer_order_number,
                    buyer_name=invoice.buyer_name,
                    buyer_gstin=invoice.buyer_gstin,
                    number_of_items=invoice.number_of_items,
                    total_amount=invoice.total_amount,
                    sgst=invoice.sgst,
                    cgst=invoice.cgst,
                )
                session.add(invoice_db)
                session.commit()

                # Create invoice items in the database
                for item in invoice.item_list:
                    invoice_item = InvoiceItemDB(
                        invoice_id=invoice_db.id,
                        description=item.description,
                        quantity=item.quantity,
                        rate_per_unit=item.rate_per_unit,
                        amount=item.amount
                    )
                    session.add(invoice_item)

                # Update processing status in the database
                status = session.query(ProcessingStatus).get(processing_status_id)
                if status:
                    status.processed_images += 1
                    session.commit()

                return invoice
            except asyncio.TimeoutError:
                print(f"Timeout processing image {extracted_text}")
                status = session.query(ProcessingStatus).get(processing_status_id)
                if status:
                    status.failed_images += 1
                    session.commit()
                return None

    except Exception as e:
        print(f"Error processing invoice: {str(e)}")
        # Update processing status to reflect failure
        status = session.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.failed_images += 1
            session.commit()
            return None

    finally:
        session.close()  # Ensure the database session is closed

async def generate_sql_query(query: str, user_id: str) -> str:
    schema = """
    Tables and relationships...
    """
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "system", "content": "Convert to SQL..."}],
        response_model=Query
    )
    return response.choices[0].message.sqlQuery

async def synthesize_response(user_question: str, results: List[Dict]) -> str:
    response = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "system", "content": "Generate response..."}]
    )
    return response.choices[0].message.content