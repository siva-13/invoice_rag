import base64
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordBearer,OAuth2PasswordRequestForm
from models import User, PDFFile
from fastapi import Depends,HTTPException,status,UploadFile,File
from database.database import get_db
from datetime import datetime, time
import shutil
from auth import get_password_hash, verify_password, create_access_token,ACCESS_TOKEN_EXPIRE_MINUTES,get_user,generate_unique_id,get_current_user
from typing import List
import os
from main import DEVICE, UPLOAD_DIR,gpu_conversion_manager





async def upload_pdfs(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Create user directory if it doesn't exist
    user_dir = os.path.join(UPLOAD_DIR, current_user.unique_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
    
    uploaded_files = []
    
    try:
        for file in files:
            # Verify if file is PDF
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a PDF"
                )
            
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{file.filename}"
            file_path = os.path.join(user_dir, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Save file info to database
            pdf_file = PDFFile(
                filename=unique_filename,
                file_path=file_path,
                user_id=current_user.unique_id
            )
            db.add(pdf_file)
            
            uploaded_files.append({
                "original_filename": file.filename,
                "saved_filename": unique_filename
            })
        
        db.commit()
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(uploaded_files)} files",
            "user_id": current_user.unique_id,
            "files": uploaded_files
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}"
        )

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
    
def delete_all_in_directory(directory):
    try:
        # Check if the directory exists
        if os.path.exists(directory):
            # Iterate over each item in the directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                # If it's a file, delete it
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                # If it's a directory, remove it and its contents
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"All contents in '{directory}' have been deleted.")
        else:
            print(f"Directory '{directory}' does not exist.")
    except Exception as e:
        print(f"Error deleting contents of '{directory}': {e}")


async def convert_pdfs_to_images_gpu(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get user's PDFs
        pdf_files = db.query(PDFFile).filter(
            PDFFile.user_id == current_user.unique_id
        ).all()
       
        if not pdf_files:
            return {
                "status": "error",
                "message": "No PDFs found for conversion"
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

def encode_image(image_path: str) -> str:
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")