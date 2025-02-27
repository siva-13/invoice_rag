from sqlalchemy.orm import Session
from database.models import Invoice, InvoiceDB, InvoiceItemDB, PDFFile, ProcessingStatus, User
from typing import Optional
from fastapi import HTTPException,status,Depends
from database.database import get_db
from services.auth import get_current_user
from database.database import SessionLocal
from config import API_SEMAPHORE,client
import asyncio
from datetime import datetime



# Add new utility function to format processing job info
def format_processing_job(status: ProcessingStatus):
    return {
        "processing_id": status.id,
        "status": status.status,
        "total_images": status.total_images,
        "processed_images": status.processed_images,
        "failed_images": status.failed_images,
        "progress_percentage": round((status.processed_images + status.failed_images) / status.total_images * 100, 2) if status.total_images > 0 else 0,
        "start_time": status.start_time,
        "end_time": status.end_time,
        "error_message": status.error_message,
        "duration": str(status.end_time - status.start_time) if status.end_time else None,
        "remaining_images": status.total_images - (status.processed_images + status.failed_images)
    }


async def process_single_image(extracted_text: str,pdf_file_id: int,user_id:int, processing_status_id: int) -> Optional[Invoice]:
    """Process a single image and store the extracted invoice data"""
    session=SessionLocal()
    try:
        # Use wait_for instead of timeout context manager
        async with API_SEMAPHORE:
            # Encode image in process pool to avoid blocking
            loop = asyncio.get_event_loop()
            try:
                # base64_image = await asyncio.wait_for(
                #     loop.run_in_executor(
                #         process_pool,
                #         encode_image,
                #         image_path
                #     ),
                #     timeout=30  # 30 second timeout for image encoding
                # )

                # Call OpenAI API with timeout
                print(extracted_text)
                if not isinstance(extracted_text, str):
                    extracted_text = str(extracted_text)
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
                                            "text": f"""
                                                    Extract structured data from this invoice image. 
                                                    Be precise and accurate in extracting the data. Check the image and extract:
                                                    Invoice Number, Seller Name, Seller GSTIN, Date of Invoice, Buyer Order Number, Buyer Name, Buyer GSTIN, 
                                                    Number of Items, Total Amount, SGST, CGST, and a list of items with Description, Quantity, Rate per Unit, and Amount.

                                                    Invoice Text: {extracted_text}
                                                    """
                                        },
                                        # {
                                        #     "type": "text",
                                        #     "text": extracted_text.strip(),
                                        # },
                                    ],
                                }
                            ],
                            response_format=Invoice,
                        )
                    ),
                    timeout=25  # 25 second timeout for API call
                )
                
                

                invoice_data = response.choices[0].message.parsed
                print(invoice_data)

                # Create invoice record in database
                invoice_db = InvoiceDB(
                    user_id=user_id,
                    pdf_file_id=pdf_file_id,
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
                print(f"Timeout processing image {extracted_text}")
                status = session.query(ProcessingStatus).get(processing_status_id)
                if status:
                    status.failed_images += 1
                    session.commit()
                return None

    # except Exception as e:
    #     print(f"Error processing image {extracted_text}: {str(e)}")
    #     status = session.query(ProcessingStatus).get(processing_status_id)
    #     if status:
    #         status.failed_images += 1
    #         session.commit()
    #     return None

    except Exception as e:
        print(f"Error processing invoice: {str(e)}")
        # Update processing status to reflect failure
        status = session.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.failed_images += 1
            session.commit()
            return None

async def process_invoices_background(
    user_id: str,
    extracted_texts: list,
    pdf_file_ids: list,
    processing_status_id: int
):
    """Background task to process invoices with extracted text."""
    db = SessionLocal()
    try:
        batch_size = 5

        for i in range(0, len(extracted_texts), batch_size):
            batch_texts = extracted_texts[i:i + batch_size]
            batch_ids = pdf_file_ids[i:i + batch_size]
            pdf_files = db.query(PDFFile).filter(PDFFile.id.in_(batch_ids)).all()

            tasks = []
            for pdf_file, text in zip(pdf_files, batch_texts):
                # Directly call the async function and append the coroutine
                task = process_single_image(
                    text,
                    pdf_file.id,
                    user_id,
                    processing_status_id
                )
                tasks.append(task)

            # Await all tasks in the batch
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0.1)  # Small delay between batches

        status = db.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.status = 'completed'
            status.end_time = datetime.utcnow()
            db.commit()

    except Exception as e:
        print(f"Background processing error: {str(e)}")
        status = db.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.status = 'failed'
            status.error_message = str(e)
            status.end_time = datetime.utcnow()
            db.commit()
    finally:
        db.close()







