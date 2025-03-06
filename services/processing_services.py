import os
import shutil
from sqlalchemy.orm import Session
from database.models import Invoice, InvoiceDB, InvoiceItemDB, PDFFile, ProcessingStatus, User
from typing import Optional
from fastapi import HTTPException,status,Depends
from database.database import get_db
from services.auth import get_current_user
from database.database import SessionLocal
from config import API_SEMAPHORE,client,PDF_IMAGE_DIR,UPLOAD_DIR
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
                # print(extracted_text)/
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
    extracted_texts: dict,  # Changed type hint to dict
    pdf_file_ids: list,
    processing_status_id: int
):
    """Background task to process invoices with extracted text."""
    db = SessionLocal()
    try:
        if not pdf_file_ids:
            raise ValueError("No PDF file IDs provided")
        
        # Assume all images belong to the first PDF (single PDF with multiple pages)
        pdf_file_id = pdf_file_ids[0]  # Single PDF ID for all images
        pdf_file = db.query(PDFFile).filter(PDFFile.id == pdf_file_id).first()
        if not pdf_file:
            raise ValueError(f"PDF file with ID {pdf_file_id} not found")
        
        user_img_dir = os.path.join(PDF_IMAGE_DIR, user_id)
        user_pdf_dir = os.path.join(UPLOAD_DIR, user_id)

        processed_img_dir = os.path.join(user_img_dir, "processed_images")
        processed_pdfs_dir = os.path.join(user_pdf_dir, "processed_pdfs")
        os.makedirs(processed_img_dir, exist_ok=True)
        os.makedirs(processed_pdfs_dir, exist_ok=True)

        # Get image paths that match the PDF filename
        image_paths = [
            os.path.join(user_img_dir, f)
            for f in os.listdir(user_img_dir)
            if f.endswith('.jpg') and pdf_file.filename.split('.')[0] in f
        ]

        successful_tasks = 0
        
        # Process each text from the dictionary
        for img_path, text in extracted_texts.items():
            # Find the corresponding full path in image_paths
            # This assumes the img_path key in extracted_texts is the basename
            img_basename = os.path.basename(img_path)
            full_img_path = None
            
            for path in image_paths:
                if os.path.basename(path) == img_basename:
                    full_img_path = path
                    break
            
            if not full_img_path:
                print(f"Warning: Could not find matching image path for {img_path}")
                continue
                
            try:
                result = await process_single_image(
                    text,
                    pdf_file.id,
                    user_id,
                    processing_status_id
                )
                
                if result is not None:
                    successful_tasks += 1
                    print(f"Successfully processed image {img_path}")
                else:
                    print(f"Failed to process image {img_path}: No result")
            except Exception as e:
                print(f"Failed to process image {img_path}: {str(e)}")
            
            # Small delay between processing
            await asyncio.sleep(0.05)

        # Move all images to processed_images directory after processing
        for image_path in image_paths:
            new_image_path = os.path.join(processed_img_dir, os.path.basename(image_path))
            if os.path.exists(image_path) and not os.path.exists(new_image_path):
                shutil.move(image_path, new_image_path)
                print(f"Moved image to {new_image_path}")
            else:
                print(f"Image move skipped: {image_path} to {new_image_path}")

        # Move the PDF file only if there were successful tasks
        if successful_tasks > 0:  
            old_pdf_path = os.path.join(user_pdf_dir, os.path.basename(pdf_file.file_path))
            new_pdf_path = os.path.join(processed_pdfs_dir, os.path.basename(pdf_file.file_path))
            if os.path.exists(old_pdf_path) and not os.path.exists(new_pdf_path):
                shutil.move(old_pdf_path, new_pdf_path)
                pdf_file.file_path = new_pdf_path   
                db.commit()
                print(f"Moved PDF to {new_pdf_path}")
            else:
                print(f"PDF move skipped: {old_pdf_path} to {new_pdf_path}")

        # Update status to completed
        status = db.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.status = 'completed'
            status.end_time = datetime.utcnow()
            db.commit()

    except Exception as e:
        print(f"Background processing error: {str(e)}")
        # traceback.print_exc()s  # Print the full traceback for debugging
        status = db.query(ProcessingStatus).get(processing_status_id)
        if status:
            status.status = 'failed'
            status.error_message = str(e)
            status.end_time = datetime.utcnow()
            db.commit()
    finally:
        db.close()