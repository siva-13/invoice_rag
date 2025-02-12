from sqlalchemy.orm import Session
from temp import process_pdfs
from main import API_SEMAPHORE,client
from database.models import Invoice, InvoiceDB, InvoiceItemDB, PDFFile, ProcessingStatus
import asyncio
from typing import Optional
import base64


async def process_single_image(image_path: str, session: Session, pdf_file: PDFFile, processing_status_id: int) -> Optional[Invoice]:
    """Process a single image and store the extracted invoice data"""
    try:
        # Use wait_for instead of timeout context manager
        async with API_SEMAPHORE:
            # Encode image in process pool to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                base64_image = await asyncio.wait_for(
                    loop.run_in_executor(
                        process_pdfs,
                        encode_image,
                        image_path
                    ),
                    timeout=30  # 30 second timeout for image encoding
                )

                # Call OpenAI API with timeout
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: client.beta.chat.completions.parse(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Extract structured data from this invoice image. Be precise and accurate in extracting the data. kindly check the image and extract the following details: Invoice Number, Seller Name, Seller GSTIN, Date of Invoice, Buyer Order Number, Buyer Name, Buyer GSTIN, Number of Items, Total Amount, SGST, CGST, and a list of items with Description, Quantity, Rate per Unit, and Amount.",
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                                        },
                                    ],
                                }
                            ],
                            response_format=Invoice,
                        )
                    ),
                    timeout=25  # 25 second timeout for API call
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
def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")