import base64
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from database.models import PDFFile, ProcessingStatus, InvoiceDB, InvoiceItemDB
from config import client, API_SEMAPHORE
import asyncio
from services.image_processing import extract_text_from_images

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

async def process_single_image(image_path: str, session: Session, pdf_file: PDFFile, processing_status_id: int) -> Optional[Invoice]:
    """Process a single image and store the extracted invoice data"""
    try:
        # Use wait_for instead of timeout context manager
        async with API_SEMAPHORE:
            # Encode image in process pool to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                extracted_texts = await asyncio.to_thread(extract_text_from_images, [image_path])
                extracted_text = extracted_texts[0] if extracted_texts else ""

                if not extracted_text.strip():
                    print(f"OCR extraction failed for {image_path}")
                    return None

                # Call OpenAI API with timeout
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: client.beta.chat.completions.parse(
                            model="deepseek-r1-distill-llama-70b",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"Extract structured data from this invoice image. Be precise and accurate in extracting the data. kindly check the image and extract the following details: Invoice Number, Seller Name, Seller GSTIN, Date of Invoice, Buyer Order Number, Buyer Name, Buyer GSTIN, Number of Items, Total Amount, SGST, CGST, and a list of items with Description, Quantity, Rate per Unit, and Amount.{extracted_text}",
                                        }
                                    ],
                                }
                            ],
                            response_format=Invoice,
                        )
                    ),
                    timeout=25  #25 second timeout for API call
                )
                
                invoice_data = response.choices[0].message.parsed

                # Create invoice record in database
                invoice_db = InvoiceDB(
                    user_id=pdf_file.user_id,
                    pdf_file_id=pdf_file.id,
                    invoice_number=invoice_data.invoice_number,
                    seller_name=invoice_data.seller_name,
                    seller_gstin=invoice_data.seller_gstin,
                    date_of_invoice=invoice_data.date_of_invoice,
                    buyer_order_number=invoice_data.buyer_order_number,
                    buyer_name=invoice_data.buyer_name,
                    buyer_gstin=invoice_data.buyer_gstin,
                    number_of_items=invoice_data.number_of_items,
                    total_amount=invoice_data.total_amount,
                    sgst=invoice_data.sgst,
                    cgst=invoice_data.cgst,
                    # raw_response=str(response)
                )
                session.add(invoice_db)
                session.flush()

                # Create invoice items
                for item in invoice_data.item_list:
                    invoice_item = InvoiceItemDB(
                        invoice_id=invoice_db.id,
                        description=item.description,
                        quantity=item.quantity,
                        rate_per_unit=item.rate_per_unit,
                        amount=item.amount
                    )
                    session.add(invoice_item)

                # Update processing status
                status = session.query(ProcessingStatus).get(processing_status_id)
                if status:
                    status.processed_images += 1
                    session.commit()

                return invoice_data

            except asyncio.TimeoutError:
                print(f"Timeout processing image {image_path}")
                status = session.query(ProcessingStatus).get(processing_status_id)
                if status:
                    status.failed_images += 1
                    session.commit()
                return None

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        status = session.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.failed_images += 1
            session.commit()
        return None


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