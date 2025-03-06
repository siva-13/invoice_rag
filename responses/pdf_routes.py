import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
import os
from sqlalchemy.orm import Session
from fastapi import Depends,UploadFile,File,HTTPException,status,APIRouter
from config import BATCH_SIZE, MAX_WORKERS, PDF_IMAGE_DIR, UPLOAD_DIR,DEVICE
from database.models import User,PDFFile
from database.database import get_db
from text import extract_text_from_images
from services.auth import get_current_user
import shutil
from datetime import datetime, time
import time
from PIL import Image
from torchvision import transforms
import torch
from paddleocr import PaddleOCR
import fitz
from services.pdf_services import GPUPDFConversionManager,GPUPDFProcessor,gpu_conversion_manager



router=APIRouter()

# PDF upload endpoint
@router.post("/upload-pdfs")
# async def upload_pdfs(
#     files: List[UploadFile] = File(...),
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     # Create user directory if it doesn't exist
#     user_dir = os.path.join(UPLOAD_DIR, current_user.unique_id)
#     if not os.path.exists(user_dir):
#         os.makedirs(user_dir)
    
#     uploaded_files = []
    
#     try:
#         for file in files:
#             # Verify if file is PDF
#             if not file.filename.lower().endswith('.pdf'):
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"File {file.filename} is not a PDF"
#                 )
            
#             # Generate unique filename
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             unique_filename = f"{timestamp}_{file.filename}"
#             file_path = os.path.join(user_dir, unique_filename)
            
#             # Save file
#             with open(file_path, "wb") as buffer:
#                 shutil.copyfileobj(file.file, buffer)
            
#             # Save file info to database
#             pdf_file = PDFFile(
#                 filename=unique_filename,
#                 file_path=file_path,
#                 user_id=current_user.unique_id
#             )
#             db.add(pdf_file)
            
#             uploaded_files.append({
#                 "original_filename": file.filename,
#                 "saved_filename": unique_filename
#             })
        
#         db.commit()
        
#         return {
#             "status": "success",
#             "message": f"Successfully uploaded {len(uploaded_files)} files",
#             "user_id": current_user.unique_id,
#             "files": uploaded_files
#         }
        
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error uploading files: {str(e)}"
#         )


async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create user directory if it doesn't exist
    user_dir = os.path.join(UPLOAD_DIR, current_user.unique_id)
    os.makedirs(user_dir, exist_ok=True)

    uploaded_files = []
    skipped_files = []

    try:
        for file in files:
            # Verify if file is a PDF
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a PDF"
                )

            # Check if the file already exists in the database
            existing_pdf = db.query(PDFFile).filter(
                PDFFile.filename == file.filename,
                PDFFile.user_id == current_user.unique_id
            ).first()

            # Check if the file already exists in the directory
            file_path = os.path.join(user_dir, file.filename)

            if existing_pdf or os.path.exists(file_path):
                skipped_files.append(file.filename)
                continue  # Skip uploading this file
            
            # Generate unique filename (optional)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{file.filename}"
            new_file_path = os.path.join(user_dir, unique_filename)

            # Save file
            with open(new_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Save file info to database
            pdf_file = PDFFile(
                filename=file.filename,  # Store original name
                file_path=new_file_path,
                user_id=current_user.unique_id
            )
            db.add(pdf_file)
            uploaded_files.append(file.filename)

        db.commit()

        response_message = {
            "status": "success",
            "message": f"Uploaded {len(uploaded_files)} files",
            "user_id": current_user.unique_id,
            "uploaded_files": uploaded_files,
        }

        if skipped_files:
            response_message["skipped_files"] = skipped_files
            response_message["skipped_message"] = "These files were already uploaded."

        return response_message

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}"
        )




# Get user's PDF files
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

@router.get("/convert-pdfs-to-images-gpu")
async def convert_pdfs_to_images_gpu(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get user's PDFs that haven't been processed yet
        # (files not in processed_pdfs folder)
        pdf_files = db.query(PDFFile).filter(
            PDFFile.user_id == current_user.unique_id,
            ~PDFFile.file_path.like("%/processed_pdfs/%")  # Exclude files in processed_pdfs folder
        ).all()
        
        if not pdf_files:
            return {
                "status": "error",
                "message": "No unprocessed PDFs found for conversion"
            }
        
        # Prepare PDF info
        pdf_info = [
            {
                'file_path': pdf.file_path,
                'filename': pdf.filename
            }
            for pdf in pdf_files
        ]
        
        # Process PDFs with GPU acceleration
        start_time = time.time()
        conversion_results = await gpu_conversion_manager.process_pdf_batch(
            pdf_info,
            current_user.unique_id
        )
        end_time = time.time()
        
        successful_conversions = [
            result for result in conversion_results
            if result['status'] == 'success'
        ]
        
        total_pages = sum(
            result['pages_converted']
            for result in successful_conversions
        )
        
        # Move successfully converted PDFs to processed_pdfs folder
        for result in successful_conversions:
            try:
                # Find the corresponding pdf_file by filename
                pdf_name = result['pdf_name']
                pdf_file = next((pdf for pdf in pdf_files if pdf.filename == pdf_name), None)
                
                if pdf_file:
                    source_path = pdf_file.file_path
                    # Get the user directory path
                    user_dir = os.path.dirname(source_path)
                    # Create the processed_pdfs directory if it doesn't exist
                    processed_dir = os.path.join(user_dir, "processed_pdfs")
                    os.makedirs(processed_dir, exist_ok=True)
                    
                    # Move the file to the processed_pdfs directory
                    destination_path = os.path.join(processed_dir, os.path.basename(source_path))
                    shutil.move(source_path, destination_path)
                    
                    # Update the file path in the database
                    pdf_file.file_path = destination_path
                    db.commit()
                    
                    # Add the new path to the result
                    result['moved_to'] = destination_path
                else:
                    result['move_status'] = "PDF file not found in database"
            except Exception as move_error:
                result['move_status'] = f"Failed to move: {str(move_error)}"
        
        return {
            "status": "success",
            "message": f"Converted {len(successful_conversions)} out of {len(pdf_files)} PDFs",
            "total_pages_converted": total_pages,
            "processing_time": f"{end_time - start_time:.2f} seconds",
            "processing_device": str(DEVICE),
            "results": conversion_results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during GPU-accelerated PDF conversion: {str(e)}"
        )  
    # async def convert_pdfs_to_images_gpu(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         # Get user's PDFs
#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()
       
#         if not pdf_files:
#             return {
#                 "status": "error",
#                 "message": "No PDFs found for conversion"
#             }
       
#         # Prepare PDF info
#         pdf_info = [
#             {
#                 'file_path': pdf.file_path,
#                 'filename': pdf.filename
#             }
#             for pdf in pdf_files
#         ]
       
#         # Process PDFs with GPU acceleration
#         start_time = time.time()
#         conversion_results = await gpu_conversion_manager.process_pdf_batch(
#             pdf_info,
#             current_user.unique_id
#         )
#         end_time = time.time()
       
#         successful_conversions = [
#             result for result in conversion_results
#             if result['status'] == 'success'
#         ]
       
#         total_pages = sum(
#             result['pages_converted']
#             for result in successful_conversions
#         )
       
#         return {
#             "status": "success",
#             "message": f"Converted {len(successful_conversions)} out of {len(pdf_files)} PDFs",
#             "total_pages_converted": total_pages,
#             "processing_time": f"{end_time - start_time:.2f} seconds",
#             "processing_device": str(DEVICE),
#             "results": conversion_results
#         }
       
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error during GPU-accelerated PDF conversion: {str(e)}"
#         )

# # async def convert_pdfs_to_images_gpu(
# #     current_user: User = Depends(get_current_user),
# #     db: Session = Depends(get_db)
# # ):
#     try:
#         # Get user's PDFs
#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()
       
#         if not pdf_files:
#             return {
#                 "status": "error",
#                 "message": "No PDFs found for conversion"
#             }
       
#         # Prepare PDF info and filter out already processed files
#         user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
#         pdf_info = []
#         for pdf in pdf_files:
#             # Check if this PDF is already in the database (processed)
#             existing_pdf = db.query(PDFFile).filter(
#                 PDFFile.filename == pdf.filename,
#                 PDFFile.user_id == current_user.unique_id
#             ).first()
            
#             # Generate expected output filename pattern
#             base_filename = os.path.splitext(pdf.filename)[0]
#             expected_image_path = os.path.join(user_image_dir, f"{base_filename}_page_1.jpg")
            
#             # Skip if PDF is in database and at least first page image exists
#             if existing_pdf and os.path.exists(expected_image_path):
#                 continue
                
#             pdf_info.append({
#                 'file_path': pdf.file_path,
#                 'filename': pdf.filename
#             })
       
#         if not pdf_info:
#             return {
#                 "status": "success",
#                 "message": "No new PDFs to process - all files already converted"
#             }
       
#         # Process PDFs with GPU acceleration
#         start_time = time.time()
#         conversion_results = await gpu_conversion_manager.process_pdf_batch(
#             pdf_info,
#             current_user.unique_id
#         )
#         end_time = time.time()
       
#         successful_conversions = [
#             result for result in conversion_results
#             if result['status'] == 'success'
#         ]
       
#         total_pages = sum(
#             result['pages_converted']
#             for result in successful_conversions
#         )
       
#         return {
#             "status": "success",
#             "message": f"Converted {len(successful_conversions)} out of {len(pdf_info)} new PDFs",
#             "total_pages_converted": total_pages,
#             "processing_time": f"{end_time - start_time:.2f} seconds",
#             "processing_device": str(DEVICE),
#             "results": conversion_results
#         }
       
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error during GPU-accelerated PDF conversion: {str(e)}"
#         )


# async def convert_pdfs_to_images_gpu(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         # Get user's PDFs
#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()
       
#         if not pdf_files:
#             return {
#                 "status": "error",
#                 "message": "No PDFs found for conversion"
#             }
       
#         # Prepare PDF info and track skipped files
#         user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
#         pdf_info = []
#         skipped_pdfs = []
        
#         for pdf in pdf_files:
#             base_filename = os.path.splitext(pdf.filename)[0]
#             expected_image_path = os.path.join(user_image_dir, f"{base_filename}_page_1.jpg")

#             # Skip if first-page image already exists
#             if os.path.exists(expected_image_path):
#                 skipped_pdfs.append(pdf.filename)
#                 continue
                
#             pdf_info.append({
#                 'file_path': pdf.file_path,
#                 'filename': pdf.filename
#             })
       
#         if not pdf_info:
#             return {
#                 "status": "success",
#                 "message": "All PDFs have already been processed.",
#                 "skipped_pdfs": skipped_pdfs
#             }
       
#         # Process PDFs with GPU acceleration
#         start_time = time.time()
#         conversion_results = await gpu_conversion_manager.process_pdf_batch(
#             pdf_info,
#             current_user.unique_id
#         )
#         end_time = time.time()
       
#         successful_conversions = [
#             result for result in conversion_results
#             if result['status'] == 'success'
#         ]
       
#         total_pages = sum(
#             result['pages_converted']
#             for result in successful_conversions
#         )
       
#         return {
#             "status": "success",
#             "message": f"Converted {len(successful_conversions)} out of {len(pdf_info)} new PDFs",
#             "total_pages_converted": total_pages,
#             "processing_time": f"{end_time - start_time:.2f} seconds",
#             "processing_device": str(DEVICE),
#             "results": conversion_results,
#             "skipped_pdfs": skipped_pdfs
#         }
       
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error during GPU-accelerated PDF conversion: {str(e)}"
#         )
