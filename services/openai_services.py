import base64
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from sqlalchemy.orm import Session
from database.models import PDFFile, ProcessingStatus, InvoiceDB, InvoiceItemDB

os.environ["OPENAI_API_KEY"] = "api-key-here"
client = OpenAI()

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
    try:
        async with API_SEMAPHORE:
            loop = asyncio.get_event_loop()
            base64_image = await loop.run_in_executor(None, encode_image, image_path)
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract structured data from this invoice image..."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }],
                    response_model=Invoice
                )
            )
            invoice_data = response.choices[0].message.content
            # ... rest of processing logic ...
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

async def generate_sql_query(query: str, user_id: str) -> str:
    schema = """
    Tables and relationships...
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Convert to SQL..."}],
        response_model=Query
    )
    return response.choices[0].message.sqlQuery

async def synthesize_response(user_question: str, results: List[Dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Generate response..."}]
    )
    return response.choices[0].message.content