# # TODO
# # Convert into modules
# # 
# # add validates to avoid same file being processed.
# #  
# from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
# from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
# from sqlalchemy.future import select
# from sqlalchemy.ext.asyncio import async_sessionmaker
# from sqlalchemy import text
# import asyncio
# from concurrent.futures import ProcessPoolExecutor
# from sqlalchemy import ForeignKey
# from passlib.context import CryptContext
# from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime, timedelta
# from typing import Optional, List, Any, Dict
# from jose import JWTError, jwt
# from pdf2image import convert_from_path
# from torchvision import transforms
# from PIL import Image
# from pydantic import BaseModel, Field
# # import aiohttp
# import json
# import base64
# from functools import partial
# import uuid
# import os
# import shutil
# import time
# import numpy as np
# import torch
# from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = "api-key-here"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Using device: {DEVICE}")

# # Constants
# PDF_IMAGE_DIR = "./pdfs_to_image"
# UPLOAD_DIR = "./uploaded_pdfs"
# MAX_WORKERS = 10  # Adjust based on your CPU cores
# BATCH_SIZE = 10  # Increased batch size for GPU processing
# RATE_LIMIT_REQUESTS = 50  # Requests per minute limit for OpenAI API
# RATE_LIMIT_WINDOW = 60  # Window in seconds

# # Update the semaphore to allow more concurrent operations
# MAX_CONCURRENT_REQUESTS = 20  # Increased from 5
# API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# # Add a process pool for CPU-bound tasks
# process_pool = ProcessPoolExecutor(max_workers=4)

# # Initialize OpenAI client
# client = OpenAI()

# # JWT Settings
# SECRET_KEY = "your-secret-key-here"  # In production, use a secure secret key
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = 100

# # Initialize FastAPI app
# app = FastAPI(title="Gen AI Invoice API")

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allow all methods
#     allow_headers=["*"],  # Allow all headers
# )

# # Database Configuration
# SQLALCHEMY_DATABASE_URL = "sqlite:///./database/users.db"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # Create async engine and session
# async_engine = create_async_engine(
#     "sqlite+aiosqlite:///./database/users.db",
#     connect_args={"check_same_thread": False}
# )
# async_session = async_sessionmaker(async_engine, expire_on_commit=False)

# # Password hashing
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # OAuth2 setup with JWT
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # Update User model to include relationships
# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     unique_id = Column(String, unique=True, index=True)
#     username = Column(String, unique=True, index=True)
#     password_hash = Column(String)
    
#     # Add relationships
#     pdf_files = relationship("PDFFile", back_populates="user")
#     invoices = relationship("InvoiceDB", back_populates="user")

# # Update PDFFile model to include relationships
# class PDFFile(Base):
#     __tablename__ = "pdf_files"

#     id = Column(Integer, primary_key=True, index=True)
#     filename = Column(String)
#     file_path = Column(String)
#     upload_time = Column(DateTime, default=datetime.utcnow)
#     user_id = Column(String, ForeignKey("users.unique_id"))
    
#     # Add relationships
#     user = relationship("User", back_populates="pdf_files")
#     invoice = relationship("InvoiceDB", back_populates="pdf_file", uselist=False)

# # Update InvoiceDB model with relationships
# class InvoiceDB(Base):
#     __tablename__ = "invoices"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, ForeignKey("users.unique_id"), index=True)
#     pdf_file_id = Column(Integer, ForeignKey("pdf_files.id"), index=True)
#     invoice_number = Column(String, index=True)
#     seller_name = Column(String)
#     seller_gstin = Column(String, nullable=True)
#     date_of_invoice = Column(String)
#     buyer_order_number = Column(String, nullable=True)
#     buyer_name = Column(String)
#     buyer_gstin = Column(String, nullable=True)
#     number_of_items = Column(Integer, nullable=True)
#     total_amount = Column(Float, nullable=True)
#     sgst = Column(Float, nullable=True)
#     cgst = Column(Float, nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     # raw_response = Column(String)  # Store the complete OpenAI response
    
#     # Add relationships
#     user = relationship("User", back_populates="invoices")
#     pdf_file = relationship("PDFFile", back_populates="invoice")
#     items = relationship("InvoiceItemDB", back_populates="invoice", cascade="all, delete-orphan")

# # Update InvoiceItemDB model with relationship
# class InvoiceItemDB(Base):
#     __tablename__ = "invoice_items"

#     id = Column(Integer, primary_key=True, index=True)
#     invoice_id = Column(Integer, ForeignKey("invoices.id"), index=True)
#     description = Column(String)
#     quantity = Column(Integer)
#     rate_per_unit = Column(Float)
#     amount = Column(Float, nullable=True)
    
#     # Add relationship
#     invoice = relationship("InvoiceDB", back_populates="items")

# # Invoice Model
# class Invoice(BaseModel):
#     # Invoice Step Model
#     class Step(BaseModel):
#         description: str = Field(..., description="Description of the item")
#         quantity: int = Field(..., description="Quantity of the item")
#         rate_per_unit: float = Field(..., description="Rate per unit of the item")
#         amount: Optional[float] = Field(None, description="Total amount for the item")
#     invoice_number: str = Field(..., description="Unique invoice identifier")
#     seller_name: str = Field(..., description="Name of the seller")
#     seller_gstin: Optional[str] = Field(None, description="GSTIN of the seller")
#     date_of_invoice: str = Field(..., description="Date of the invoice in YYYY-MM-DD format")
#     buyer_order_number: Optional[str] = Field(None, description="Order number from the buyer")
#     buyer_name: str = Field(..., description="Name of the buyer")
#     buyer_gstin: Optional[str] = Field(None, description="GSTIN of the buyer")
#     number_of_items: Optional[int] = Field(None, description="Number of items in the order")
#     item_list: list[Step]
#     total_amount: Optional[float] = Field(None, description="Total amount for the invoice")
#     sgst: Optional[float] = Field(None, description="State GST amount")
#     cgst: Optional[float] = Field(None, description="Central GST amount")

# class Query(BaseModel):
#     sqlQuery: str = Field(..., description="SQL query based on user prompt")

# class ProcessingStatus(Base):
#     __tablename__ = "processing_status"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(String, ForeignKey("users.unique_id"))
#     start_time = Column(DateTime, default=datetime.utcnow)
#     end_time = Column(DateTime, nullable=True)
#     total_images = Column(Integer)
#     processed_images = Column(Integer, default=0)
#     failed_images = Column(Integer, default=0)
#     status = Column(String)  # 'processing', 'completed', 'failed'
#     error_message = Column(String, nullable=True)
    
# # Create tables
# Base.metadata.create_all(bind=engine)

# # Create directory for uploaded PDFs
# if not os.path.exists(UPLOAD_DIR):
#     os.makedirs(UPLOAD_DIR)

# class GPUPDFProcessor:
#     def __init__(self):
#         self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
#         self.device = DEVICE
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ConvertImageDtype(torch.float32)
#         ])
       
#     async def convert_pdf_to_images(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
#         try:
#             convert_func = partial(
#                 self._convert_single_pdf,
#                 output_dir=output_dir,
#                 filename=filename
#             )
           
#             loop = asyncio.get_event_loop()
#             result = await loop.run_in_executor(
#                 self.executor,
#                 convert_func,
#                 pdf_path
#             )
#             return result
           
#         except Exception as e:
#             print(f"Error converting PDF {filename}: {str(e)}")
#             return []

#     def process_image_batch_gpu(self, images: List[Image.Image]) -> List[Image.Image]:
#         """Process a batch of images using GPU, but retain their original appearance."""
#         # Convert images to tensors and move to GPU without altering appearance
#         tensors = [self.transform(img) for img in images]
#         batch = torch.stack(tensors).to(self.device)

#         # No processing like contrast enhancement, just convert back to CPU and PIL Images
#         batch = batch.cpu()
#         processed_images = [
#             transforms.ToPILImage()(img)
#             for img in batch
#         ]
       
#         return processed_images

#     def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
#         """Convert PDF to images with GPU acceleration without altering appearance."""
#         try:
#             # Convert PDF pages to images
#             images = convert_from_path(
#                 pdf_path,
#                 dpi=200,
#                 fmt='jpeg',
#                 thread_count=2
#             )
           
#             saved_paths = []
#             base_filename = os.path.splitext(filename)[0]
           
#             # Process images in batches using GPU
#             for i in range(0, len(images), BATCH_SIZE):
#                 batch = images[i:i + BATCH_SIZE]
#                 processed_batch = self.process_image_batch_gpu(batch)
               
#                 # Save processed images
#                 for j, processed_img in enumerate(processed_batch):
#                     page_num = i + j + 1
#                     image_path = os.path.join(
#                         output_dir,
#                         f"{base_filename}_page_{page_num}.jpg"
#                     )
#                     processed_img.save(
#                         image_path,
#                         'JPEG',
#                         quality=90,
#                         optimize=True
#                     )
#                     saved_paths.append(image_path)
           
#             return saved_paths
           
#         except Exception as e:
#             print(f"Error in _convert_single_pdf: {str(e)}")
#             return []

# class GPUPDFConversionManager:
#     def __init__(self):
#         self.processor = GPUPDFProcessor()
       
#     async def process_pdf_batch(
#         self,
#         pdf_files: List[dict],
#         user_id: str
#     ) -> List[dict]:
#         # Create user directory
#         user_image_dir = os.path.join(PDF_IMAGE_DIR, user_id)
#         os.makedirs(user_image_dir, exist_ok=True)
       
#         # Process PDFs in optimized batches
#         results = []
#         for i in range(0, len(pdf_files), BATCH_SIZE):
#             batch = pdf_files[i:i + BATCH_SIZE]
#             tasks = [
#                 self.processor.convert_pdf_to_images(
#                     pdf['file_path'],
#                     user_image_dir,
#                     pdf['filename']
#                 )
#                 for pdf in batch
#             ]
           
#             batch_results = await asyncio.gather(*tasks)
           
#             for pdf, image_paths in zip(batch, batch_results):
#                 results.append({
#                     'pdf_name': pdf['filename'],
#                     'image_paths': image_paths,
#                     'status': 'success' if image_paths else 'failed',
#                     'pages_converted': len(image_paths)
#                 })
               
#         return results

# # Initialize GPU-enabled manager
# gpu_conversion_manager = GPUPDFConversionManager()
    
# # Database dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Helper functions
# def generate_unique_id():
#     """Generate a unique ID based on timestamp and UUID"""
#     timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
#     random_string = str(uuid.uuid4())[:8]
#     return f"USER_{timestamp}_{random_string}"

# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password):
#     return pwd_context.hash(password)

# def get_user(db: Session, username: str):
#     return db.query(User).filter(User.username == username).first()

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except JWTError:
#         raise credentials_exception
        
#     user = get_user(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

# # API Endpoints
# @app.post("/signup")
# async def signup(
#     form_data: OAuth2PasswordRequestForm = Depends(),
#     db: Session = Depends(get_db)
# ):
#     # Check if user already exists
#     if get_user(db, form_data.username):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username already registered"
#         )
    
#     # Create new user
#     unique_id = generate_unique_id()
#     new_user = User(
#         username=form_data.username,
#         password_hash=get_password_hash(form_data.password),
#         unique_id=unique_id
#     )
    
#     try:
#         db.add(new_user)
#         db.commit()
#         db.refresh(new_user)
        
#         # Create access token
#         access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#         access_token = create_access_token(
#             data={"sub": new_user.username}, 
#             expires_delta=access_token_expires
#         )
        
#         return {
#             "message": "User created successfully",
#             "username": new_user.username,
#             "unique_id": new_user.unique_id,
#             "access_token": access_token,
#             "token_type": "bearer"
#         }
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error creating user: {str(e)}"
#         )

# @app.post("/token")
# async def login(
#     form_data: OAuth2PasswordRequestForm = Depends(),
#     db: Session = Depends(get_db)
# ):
#     # Get user from database
#     user = get_user(db, form_data.username)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     # Verify password
#     if not verify_password(form_data.password, user.password_hash):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     # Create access token
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, 
#         expires_delta=access_token_expires
#     )
    
#     return {
#         "access_token": access_token,
#         "token_type": "bearer",
#         "username": user.username,
#         "unique_id": user.unique_id
#     }

# @app.get("/users/me")
# async def get_user_info(current_user: User = Depends(get_current_user)):
#     return {
#         "username": current_user.username,
#         "unique_id": current_user.unique_id
#     }

# # PDF upload endpoint
# @app.post("/upload-pdfs")
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

# # Get user's PDF files
# @app.get("/my-pdfs")
# async def get_user_pdfs(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()
        
#         return {
#             "status": "success",
#             "user_id": current_user.unique_id,
#             "files": [
#                 {
#                     "filename": pdf.filename,
#                     "upload_time": pdf.upload_time
#                 } for pdf in pdf_files
#             ]
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving PDF files: {str(e)}"
#         )
    
# # delete all files in directory
# def delete_all_in_directory(directory):
#     try:
#         # Check if the directory exists
#         if os.path.exists(directory):
#             # Iterate over each item in the directory
#             for item in os.listdir(directory):
#                 item_path = os.path.join(directory, item)
#                 # If it's a file, delete it
#                 if os.path.isfile(item_path) or os.path.islink(item_path):
#                     os.unlink(item_path)
#                 # If it's a directory, remove it and its contents
#                 elif os.path.isdir(item_path):
#                     shutil.rmtree(item_path)
#             print(f"All contents in '{directory}' have been deleted.")
#         else:
#             print(f"Directory '{directory}' does not exist.")
#     except Exception as e:
#         print(f"Error deleting contents of '{directory}': {e}")

# # Only on development
# @app.get("/drop-tables")
# async def delete_table(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
#     try:
#         Base.metadata.drop_all(bind=engine)
#         Base.metadata.create_all(bind=engine)
#         delete_all_in_directory(UPLOAD_DIR)
#         delete_all_in_directory(PDF_IMAGE_DIR)
#         return {"status": "success", "message": "Tables dropped successfully"}
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error dropping tables: {str(e)}"
#         )
    
# # Updated endpoint for GPU-accelerated conversion
# @app.get("/convert-pdfs-to-images-gpu")
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

# def encode_image(image_path: str) -> str:
#     """Encode image to base64 string"""
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode("utf-8")

# # async def process_image_with_openai(image_path: str) -> Invoice:
# #     """Process an image using OpenAI's Vision API asynchronously"""
# #     try:
# #         # Initialize OpenAI client
# #         client = OpenAI(api_key=openai_api_key)
        
# #         # Run the synchronous OpenAI call in a thread pool
# #         loop = asyncio.get_event_loop()
# #         invoice = await loop.run_in_executor(
# #             None,
# #             partial(process_with_openai_sync, image_path, client)
# #         )
        
# #         return invoice
        
# #     except Exception as e:
# #         print(f"Error processing image with OpenAI: {str(e)}")
# #         raise

# #  revsion 2 ----------------------------------------------------------------------------------------------------------------------------
# async def process_single_image(image_path: str, session: Session, pdf_file: PDFFile, processing_status_id: int) -> Optional[Invoice]:
#     """Process a single image and store the extracted invoice data"""
#     try:
#         # Use wait_for instead of timeout context manager
#         async with API_SEMAPHORE:
#             # Encode image in process pool to avoid blocking
#             loop = asyncio.get_event_loop()
#             try:
#                 base64_image = await asyncio.wait_for(
#                     loop.run_in_executor(
#                         process_pool,
#                         encode_image,
#                         image_path
#                     ),
#                     timeout=30  # 30 second timeout for image encoding
#                 )

#                 # Call OpenAI API with timeout
#                 response = await asyncio.wait_for(
#                     loop.run_in_executor(
#                         None,
#                         lambda: client.beta.chat.completions.parse(
#                             model="gpt-4o",
#                             messages=[
#                                 {
#                                     "role": "user",
#                                     "content": [
#                                         {
#                                             "type": "text",
#                                             "text": "Extract structured data from this invoice image. Be precise and accurate in extracting the data. kindly check the image and extract the following details: Invoice Number, Seller Name, Seller GSTIN, Date of Invoice, Buyer Order Number, Buyer Name, Buyer GSTIN, Number of Items, Total Amount, SGST, CGST, and a list of items with Description, Quantity, Rate per Unit, and Amount.",
#                                         },
#                                         {
#                                             "type": "image_url",
#                                             "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
#                                         },
#                                     ],
#                                 }
#                             ],
#                             response_format=Invoice,
#                         )
#                     ),
#                     timeout=25  # 25 second timeout for API call
#                 )
                
#                 invoice_data = response.choices[0].message.parsed

#                 # Create invoice record in database
#                 invoice_db = InvoiceDB(
#                     user_id=pdf_file.user_id,
#                     pdf_file_id=pdf_file.id,
#                     invoice_number=invoice_data.invoice_number,
#                     seller_name=invoice_data.seller_name,
#                     seller_gstin=invoice_data.seller_gstin,
#                     date_of_invoice=invoice_data.date_of_invoice,
#                     buyer_order_number=invoice_data.buyer_order_number,
#                     buyer_name=invoice_data.buyer_name,
#                     buyer_gstin=invoice_data.buyer_gstin,
#                     number_of_items=invoice_data.number_of_items,
#                     total_amount=invoice_data.total_amount,
#                     sgst=invoice_data.sgst,
#                     cgst=invoice_data.cgst,
#                     # raw_response=str(response)
#                 )
#                 session.add(invoice_db)
#                 session.flush()

#                 # Create invoice items
#                 for item in invoice_data.item_list:
#                     invoice_item = InvoiceItemDB(
#                         invoice_id=invoice_db.id,
#                         description=item.description,
#                         quantity=item.quantity,
#                         rate_per_unit=item.rate_per_unit,
#                         amount=item.amount
#                     )
#                     session.add(invoice_item)

#                 # Update processing status
#                 status = session.query(ProcessingStatus).get(processing_status_id)
#                 if status:
#                     status.processed_images += 1
#                     session.commit()

#                 return invoice_data

#             except asyncio.TimeoutError:
#                 print(f"Timeout processing image {image_path}")
#                 status = session.query(ProcessingStatus).get(processing_status_id)
#                 if status:
#                     status.failed_images += 1
#                     session.commit()
#                 return None

#     except Exception as e:
#         print(f"Error processing image {image_path}: {str(e)}")
#         status = session.query(ProcessingStatus).get(processing_status_id)
#         if status:
#             status.failed_images += 1
#             session.commit()
#         return None

# async def process_invoices_background(
#     user_id: str,
#     image_paths: list,
#     pdf_file_ids: list,
#     processing_status_id: int
# ):
#     """Background task to process invoices with improved concurrency"""
#     db = SessionLocal()
#     try:
#         # Process images in smaller batches to maintain responsiveness
#         batch_size = 5
#         for i in range(0, len(image_paths), batch_size):
#             batch = image_paths[i:i + batch_size]
            
#             # Get PDF files for this batch
#             pdf_files = db.query(PDFFile).filter(PDFFile.id.in_(pdf_file_ids)).all()
            
#             tasks = []
#             for pdf_file in pdf_files:
#                 base_filename = os.path.splitext(pdf_file.filename)[0]
#                 pdf_images = [
#                     path for path in batch 
#                     if os.path.basename(path).startswith(base_filename)
#                 ]
                
#                 for image_path in pdf_images:
#                     task = process_single_image(
#                         image_path,
#                         db,
#                         pdf_file,
#                         processing_status_id
#                     )
#                     tasks.append(task)

#             # Process batch with gather
#             await asyncio.gather(*tasks, return_exceptions=True)
            
#             # Small delay between batches to prevent overload
#             await asyncio.sleep(0.1)

#         # Update final status
#         status = db.query(ProcessingStatus).get(processing_status_id)
#         if status:
#             status.status = 'completed'
#             status.end_time = datetime.utcnow()
#             db.commit()

#     except Exception as e:
#         status = db.query(ProcessingStatus).get(processing_status_id)
#         if status:
#             status.status = 'failed'
#             status.error_message = str(e)
#             status.end_time = datetime.utcnow()
#             db.commit()
#     finally:
#         db.close()

# # Update the route to use the improved background processing
# @app.post("/process-invoices")
# async def process_invoices(
#     background_tasks: BackgroundTasks,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         # Get all image paths for the user
#         user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
#         if not os.path.exists(user_image_dir):
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No images found for processing"
#             )

#         # Get all PDF files for the user
#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()

#         if not pdf_files:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No PDF files found"
#             )

#         # Get PDF file IDs
#         pdf_file_ids = [pdf.id for pdf in pdf_files]

#         # Get all relevant image paths
#         image_paths = [
#             os.path.join(user_image_dir, f) 
#             for f in os.listdir(user_image_dir) 
#             if f.endswith('.jpg')
#         ]

#         if not image_paths:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No images found for processing"
#             )

#         # Create processing status record
#         processing_status = ProcessingStatus(
#             user_id=current_user.unique_id,
#             total_images=len(image_paths),
#             status='processing'
#         )
#         db.add(processing_status)
#         db.commit()
#         db.refresh(processing_status)

#         # Start background processing
#         background_tasks.add_task(
#             process_invoices_background,
#             current_user.unique_id,
#             image_paths,
#             pdf_file_ids,
#             processing_status.id
#         )

#         return {
#             "status": "processing_started",
#             "message": "Invoice processing started in background",
#             "total_images": len(image_paths),
#             "processing_id": processing_status.id
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error starting invoice processing: {str(e)}"
#         )
    
# @app.post("/process-invoices")
# async def process_invoices(
#     background_tasks: BackgroundTasks,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         # Get all image paths for the user
#         user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
#         if not os.path.exists(user_image_dir):
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No images found for processing"
#             )

#         # Get all PDF files for the user
#         pdf_files = db.query(PDFFile).filter(
#             PDFFile.user_id == current_user.unique_id
#         ).all()

#         if not pdf_files:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No PDF files found"
#             )

#         # Get PDF file IDs
#         pdf_file_ids = [pdf.id for pdf in pdf_files]

#         # Get all relevant image paths
#         image_paths = [
#             os.path.join(user_image_dir, f) 
#             for f in os.listdir(user_image_dir) 
#             if f.endswith('.jpg')
#         ]

#         if not image_paths:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="No images found for processing"
#             )

#         # Create processing status record
#         processing_status = ProcessingStatus(
#             user_id=current_user.unique_id,
#             total_images=len(image_paths),
#             status='processing'
#         )
#         db.add(processing_status)
#         db.commit()
#         db.refresh(processing_status)

#         # Start background processing with IDs instead of PDF objects
#         background_tasks.add_task(
#             process_invoices_background,
#             current_user.unique_id,
#             image_paths,
#             pdf_file_ids,  # Pass IDs instead of PDF objects
#             processing_status.id
#         )

#         return {
#             "status": "processing_started",
#             "message": "Invoice processing started in background",
#             "total_images": len(image_paths),
#             "processing_id": processing_status.id
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error starting invoice processing: {str(e)}"
#         )

# @app.get("/processing-status/{processing_id}")
# async def get_processing_status(
#     processing_id: int,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get the status of a processing job"""
#     status = db.query(ProcessingStatus).filter(
#         ProcessingStatus.id == processing_id,
#         ProcessingStatus.user_id == current_user.unique_id
#     ).first()

#     if not status:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail="Processing job not found"
#         )

#     return {
#         "status": status.status,
#         "total_images": status.total_images,
#         "processed_images": status.processed_images,
#         "failed_images": status.failed_images,
#         "start_time": status.start_time,
#         "end_time": status.end_time,
#         "error_message": status.error_message
#     }


# # Add new utility function to format processing job info
# def format_processing_job(status: ProcessingStatus):
#     return {
#         "processing_id": status.id,
#         "status": status.status,
#         "total_images": status.total_images,
#         "processed_images": status.processed_images,
#         "failed_images": status.failed_images,
#         "progress_percentage": round((status.processed_images + status.failed_images) / status.total_images * 100, 2) if status.total_images > 0 else 0,
#         "start_time": status.start_time,
#         "end_time": status.end_time,
#         "error_message": status.error_message,
#         "duration": str(status.end_time - status.start_time) if status.end_time else None,
#         "remaining_images": status.total_images - (status.processed_images + status.failed_images)
#     }



# @app.get("/processing-jobs")
# async def get_all_processing_jobs(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
#     skip: int = 0,
#     limit: int = 100,
#     status_filter: Optional[str] = None
# ):
#     """
#     Get all processing jobs for the current user with optional filtering and pagination
#     """
#     try:
#         # Base query
#         query = db.query(ProcessingStatus).filter(
#             ProcessingStatus.user_id == current_user.unique_id
#         )
        
#         # Apply status filter if provided
#         if status_filter:
#             query = query.filter(ProcessingStatus.status == status_filter)
        
#         # Get total count for pagination
#         total_jobs = query.count()
        
#         # Get jobs with pagination and ordering
#         jobs = query.order_by(ProcessingStatus.start_time.desc())\
#                    .offset(skip)\
#                    .limit(limit)\
#                    .all()

#         # Format response with summary statistics
#         active_jobs = sum(1 for job in jobs if job.status == 'processing')
#         completed_jobs = sum(1 for job in jobs if job.status == 'completed')
#         failed_jobs = sum(1 for job in jobs if job.status == 'failed')
        
#         total_images_processed = sum(job.processed_images for job in jobs)
#         total_images_failed = sum(job.failed_images for job in jobs)
        
#         return {
#             "jobs": [format_processing_job(job) for job in jobs],
#             "pagination": {
#                 "total": total_jobs,
#                 "skip": skip,
#                 "limit": limit,
#                 "has_more": (skip + limit) < total_jobs
#             },
#             "summary": {
#                 "total_jobs": total_jobs,
#                 "active_jobs": active_jobs,
#                 "completed_jobs": completed_jobs,
#                 "failed_jobs": failed_jobs,
#                 "total_images_processed": total_images_processed,
#                 "total_images_failed": total_images_failed
#             }
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving processing jobs: {str(e)}"
#         )

# @app.get("/processing-jobs/active")
# async def get_active_processing_jobs(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Get only active processing jobs for the current user
#     """
#     try:
#         active_jobs = db.query(ProcessingStatus).filter(
#             ProcessingStatus.user_id == current_user.unique_id,
#             ProcessingStatus.status == 'processing'
#         ).order_by(ProcessingStatus.start_time.desc()).all()

#         return {
#             "active_jobs": [format_processing_job(job) for job in active_jobs],
#             "count": len(active_jobs)
#         }

#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving active jobs: {str(e)}"
#         )

# @app.get("/processing-jobs/summary")
# async def get_processing_jobs_summary(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Get a summary of all processing jobs for the current user
#     Returns both summary statistics and a list of all formatted jobs
#     """
#     try:
#         # Get all jobs for the user
#         jobs = db.query(ProcessingStatus).filter(
#             ProcessingStatus.user_id == current_user.unique_id
#         ).all()
        
#         # Calculate statistics
#         total_jobs = len(jobs)
#         status_counts = {
#             'processing': 0,
#             'completed': 0,
#             'failed': 0
#         }
#         total_images = 0
#         total_processed = 0
#         total_failed = 0
        
#         for job in jobs:
#             status_counts[job.status] += 1
#             total_images += job.total_images
#             total_processed += job.processed_images
#             total_failed += job.failed_images
        
#         # Calculate success rate
#         success_rate = (total_processed / total_images * 100) if total_images > 0 else 0
        
#         # Format all jobs
#         formatted_jobs = [format_processing_job(job) for job in jobs]
        
#         return {
#             "total_jobs": total_jobs,
#             "status_breakdown": status_counts,
#             "image_statistics": {
#                 "total_images": total_images,
#                 "processed_images": total_processed,
#                 "failed_images": total_failed,
#                 "success_rate": round(success_rate, 2)
#             },
#             "jobs": formatted_jobs  # Return all formatted jobs instead of just the latest
#         }
    
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving jobs summary: {str(e)}"
#         )

# @app.get("/invoices")
# async def get_user_invoices(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
#     skip: int = 0,
#     limit: int = 50,
#     sort_by: Optional[str] = "created_at",
#     sort_order: Optional[str] = "desc"
# ):
#     """
#     Retrieve all invoices for the current user with pagination and sorting options
#     """
#     try:
#         # Base query
#         query = db.query(InvoiceDB).filter(
#             InvoiceDB.user_id == current_user.unique_id
#         )
        
#         # Apply sorting
#         sort_field = getattr(InvoiceDB, sort_by, InvoiceDB.created_at)
#         if sort_order.lower() == "desc":
#             query = query.order_by(sort_field.desc())
#         else:
#             query = query.order_by(sort_field.asc())
            
#         # Get total count for pagination
#         total_invoices = query.count()
        
#         # Apply pagination
#         invoices = query.offset(skip).limit(limit).all()
        
#         # Format response with detailed invoice information
#         formatted_invoices = []
#         for invoice in invoices:
#             # Get all items for this invoice
#             items = db.query(InvoiceItemDB).filter(
#                 InvoiceItemDB.invoice_id == invoice.id
#             ).all()
            
#             formatted_items = [
#                 {
#                     "description": item.description,
#                     "quantity": item.quantity,
#                     "rate_per_unit": item.rate_per_unit,
#                     "amount": item.amount
#                 }
#                 for item in items
#             ]
            
#             # Get associated PDF file information
#             pdf_file = db.query(PDFFile).filter(
#                 PDFFile.id == invoice.pdf_file_id
#             ).first()
            
#             formatted_invoices.append({
#                 "id": invoice.id,
#                 "invoice_number": invoice.invoice_number,
#                 "seller_name": invoice.seller_name,
#                 "seller_gstin": invoice.seller_gstin,
#                 "date_of_invoice": invoice.date_of_invoice,
#                 "buyer_order_number": invoice.buyer_order_number,
#                 "buyer_name": invoice.buyer_name,
#                 "buyer_gstin": invoice.buyer_gstin,
#                 "number_of_items": invoice.number_of_items,
#                 "total_amount": invoice.total_amount,
#                 "sgst": invoice.sgst,
#                 "cgst": invoice.cgst,
#                 "created_at": invoice.created_at,
#                 "items": formatted_items,
#                 "pdf_info": {
#                     "filename": pdf_file.filename if pdf_file else None,
#                     "upload_time": pdf_file.upload_time if pdf_file else None
#                 }
#             })
        
#         # Calculate summary statistics
#         total_amount = sum(invoice.total_amount or 0 for invoice in invoices)
#         total_items = sum(invoice.number_of_items or 0 for invoice in invoices)
        
#         return {
#             "invoices": formatted_invoices,
#             "pagination": {
#                 "total": total_invoices,
#                 "skip": skip,
#                 "limit": limit,
#                 "has_more": (skip + limit) < total_invoices
#             },
#             "summary": {
#                 "total_invoices": total_invoices,
#                 "total_amount": total_amount,
#                 "total_items": total_items,
#                 "average_amount": total_amount / len(invoices) if invoices else 0
#             }
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving invoices: {str(e)}"
#         )

# @app.get("/invoices/{invoice_id}")
# async def get_invoice_detail(
#     invoice_id: int,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Retrieve detailed information for a specific invoice
#     """
#     try:
#         # Get invoice with verification of ownership
#         invoice = db.query(InvoiceDB).filter(
#             InvoiceDB.id == invoice_id,
#             InvoiceDB.user_id == current_user.unique_id
#         ).first()
        
#         if not invoice:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail="Invoice not found"
#             )
        
#         # Get invoice items
#         items = db.query(InvoiceItemDB).filter(
#             InvoiceItemDB.invoice_id == invoice.id
#         ).all()
        
#         # Get associated PDF file
#         pdf_file = db.query(PDFFile).filter(
#             PDFFile.id == invoice.pdf_file_id
#         ).first()
        
#         return {
#             "invoice_details": {
#                 "id": invoice.id,
#                 "invoice_number": invoice.invoice_number,
#                 "seller_name": invoice.seller_name,
#                 "seller_gstin": invoice.seller_gstin,
#                 "date_of_invoice": invoice.date_of_invoice,
#                 "buyer_order_number": invoice.buyer_order_number,
#                 "buyer_name": invoice.buyer_name,
#                 "buyer_gstin": invoice.buyer_gstin,
#                 "number_of_items": invoice.number_of_items,
#                 "total_amount": invoice.total_amount,
#                 "sgst": invoice.sgst,
#                 "cgst": invoice.cgst,
#                 "created_at": invoice.created_at
#             },
#             "items": [
#                 {
#                     "description": item.description,
#                     "quantity": item.quantity,
#                     "rate_per_unit": item.rate_per_unit,
#                     "amount": item.amount
#                 }
#                 for item in items
#             ],
#             "pdf_info": {
#                 "filename": pdf_file.filename if pdf_file else None,
#                 "upload_time": pdf_file.upload_time if pdf_file else None
#             }
#         }
        
#     except HTTPException as he:
#         raise he
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error retrieving invoice details: {str(e)}"
#         )
    
# async def generate_sql_query(query: str, client: OpenAI, user_id: str) -> str:
#     """Convert natural language query to SQL using OpenAI"""

#     schema = """
#     Tables:
#     - invoices: id, user_id, invoice_number, seller_name, seller_gstin, date_of_invoice, 
#                 buyer_order_number, buyer_name, buyer_gstin, number_of_items, 
#                 total_amount, sgst, cgst, created_at
#     - invoice_items: id, invoice_id, description, quantity, rate_per_unit, amount
    
#     Relationships:
#     - invoices has many invoice_items (one-to-many)
#     - Both tables are filtered by user_id for security
#     """

#     response = await asyncio.get_running_loop().run_in_executor(
#         None,
#         lambda: client.beta.chat.completions.parse(
#         model="gpt-4o-mini",
#         messages=[
#             {
#                 "role": "system",
#                 "content": f"You are a SQL expert. Convert natural language queries to SQL based on this schema:\n{schema}\nOnly return the SQL query, nothing else."
#             },
#             {
#                 "role": "user",
#                 "content": f"Convert this question to SQL (always include user_id filter: {user_id}): {query}"
#             }
#         ],
#         response_format=Query,
#     )
#     )
    
#     invoice_data = response.choices[0].message.parsed
#     print(invoice_data.sqlQuery)
#     return invoice_data.sqlQuery

# async def synthesize_response(user_question: str, sql_query_results: List[Dict[str, Any]], client: OpenAI) -> str:
#     """Convert SQL results to natural language response using OpenAI"""
#     response = await asyncio.get_running_loop().run_in_executor(
#         None,
#         lambda: client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Please generate a clear and concise human-readable response based on the user's question and the SQL query results. If no results are found, explain this to the user."},
#                 {"role": "user", "content": f"User's Question: {user_question} SQL Query Results: {sql_query_results}."}
#             ],
#             temperature=0.1
#         )
#     )
    
#     return response.choices[0].message.content.strip()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/query-invoices")
# async def query_invoices(
#     query: QueryRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Process natural language queries about invoices and return results
#     Example queries:
#     - "What's my total invoice amount for last month?"
#     - "Show me all invoices from seller X"
#     - "What's the average invoice amount?"
#     """
#     try:
#         print(query)
#         # Generate SQL query from natural language
#         sql_query = await generate_sql_query(query, client, current_user.unique_id)
        
#         # Add user_id filter if not present (security measure)
#         if "WHERE" not in sql_query.upper():
#             sql_query = f"{sql_query} WHERE user_id = {current_user.unique_id}"
#         elif "user_id" not in sql_query:
#             sql_query = sql_query.replace("WHERE", f"WHERE user_id = {current_user.unique_id} AND")
            
#         # Execute query with parameters
#         result = await asyncio.get_running_loop().run_in_executor(
#             None,
#             lambda: db.execute(
#                 text(sql_query),
#                 {"user_id": current_user.unique_id}
#             ).fetchall()
#         )
#         print(f"sql response: {result}")
#         # Convert result to list of dictionaries
#         results = [
#             {column: value for column, value in zip(row._mapping.keys(), row._mapping.values())}
#             for row in result
#         ]
        
#         # Synthesize natural language response
#         explanation = await synthesize_response(query, results, client)
        
#         return {
#             "query": query,
#             "sql_query": sql_query,
#             "results": results,
#             "explanation": explanation
#         }
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Error processing query: {str(e)}"
#         )



# TODO
# Convert into modules
# 
# add validates to avoid same file being processed.
#  
from fastapi import FastAPI, HTTPException, Depends, Query, status, File, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import Engine, create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import text
import asyncio
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import ForeignKey
from passlib.context import CryptContext  # type: ignore
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, List, Any, Dict
from jose import JWTError, jwt
from pdf2image import convert_from_path
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel, Field
# import aiohttp
import json
import base64
from functools import partial
import uuid
import os
import shutil
import time
import numpy as np
import torch
from openai import OpenAI
from database.models import Invoice, InvoiceDB, InvoiceItemDB, PDFFile, ProcessingStatus, SessionLocal, User,Base
from database.database import get_db
from services.auth import get_current_user,generate_unique_id,get_password_hash,create_access_token,get_user,ACCESS_TOKEN_EXPIRE_MINUTES, verify_password
from response.response import router


os.environ["OPENAI_API_KEY"] = "api-key-here"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# Constants
PDF_IMAGE_DIR = "./pdfs_to_image"
UPLOAD_DIR = "./uploaded_pdfs"
MAX_WORKERS = 10  # Adjust based on your CPU cores
BATCH_SIZE = 10  # Increased batch size for GPU processing
RATE_LIMIT_REQUESTS = 50  # Requests per minute limit for OpenAI API
RATE_LIMIT_WINDOW = 60  # Window in seconds

# Update the semaphore to allow more concurrent operations
MAX_CONCURRENT_REQUESTS = 20  # Increased from 5
API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Add a process pool for CPU-bound tasks
process_pool = ProcessPoolExecutor(max_workers=4)

# Initialize OpenAI client
client = OpenAI()



# Initialize FastAPI app
app = FastAPI(title="Gen AI Invoice API")
app.include_router(router)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)



# Create async engine and session
async_engine = create_async_engine(
    "sqlite+aiosqlite:///./database/users.db",
    connect_args={"check_same_thread": False}
)
async_session = async_sessionmaker(async_engine, expire_on_commit=False)



# Create directory for uploaded PDFs
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class GPUPDFProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.device = DEVICE
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32)
        ])
      
    async def convert_pdf_to_images(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
        try:
            convert_func = partial(
                self._convert_single_pdf,
                output_dir=output_dir,
                filename=filename
            )
           
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                convert_func,
                pdf_path
            )
            return result
           
        except Exception as e:
            print(f"Error converting PDF {filename}: {str(e)}")
            return []

    def process_image_batch_gpu(self, images: List[Image.Image]) -> List[Image.Image]:
        """Process a batch of images using GPU, but retain their original appearance."""
        # Convert images to tensors and move to GPU without altering appearance
        tensors = [self.transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)

        # No processing like contrast enhancement, just convert back to CPU and PIL Images
        batch = batch.cpu()
        processed_images = [
            transforms.ToPILImage()(img)
            for img in batch
        ]
       
        return processed_images

    def _convert_single_pdf(self, pdf_path: str, output_dir: str, filename: str) -> List[str]:
        """Convert PDF to images with GPU acceleration without altering appearance."""
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=200,
                fmt='jpeg',
                thread_count=2
            )
           
            saved_paths = []
            base_filename = os.path.splitext(filename)[0]
           
            # Process images in batches using GPU
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i:i + BATCH_SIZE]
                processed_batch = self.process_image_batch_gpu(batch)
               
                # Save processed images
                for j, processed_img in enumerate(processed_batch):
                    page_num = i + j + 1
                    image_path = os.path.join(
                        output_dir,
                        f"{base_filename}_page_{page_num}.jpg"
                    )
                    processed_img.save(
                        image_path,
                        'JPEG',
                        quality=90,
                        optimize=True
                    )
                    saved_paths.append(image_path)
           
            return saved_paths
           
        except Exception as e:
            print(f"Error in _convert_single_pdf: {str(e)}")
            return []

class GPUPDFConversionManager:
    def __init__(self):
        self.processor = GPUPDFProcessor()
       
    async def process_pdf_batch(
        self,
        pdf_files: List[dict],
        user_id: str
    ) -> List[dict]:
        # Create user directory
        user_image_dir = os.path.join(PDF_IMAGE_DIR, user_id)
        os.makedirs(user_image_dir, exist_ok=True)
       
        # Process PDFs in optimized batches
        results = []
        for i in range(0, len(pdf_files), BATCH_SIZE):
            batch = pdf_files[i:i + BATCH_SIZE]
            tasks = [
                self.processor.convert_pdf_to_images(
                    pdf['file_path'],
                    user_image_dir,
                    pdf['filename']
                )
                for pdf in batch
            ]
           
            batch_results = await asyncio.gather(*tasks)
           
            for pdf, image_paths in zip(batch, batch_results):
                results.append({
                    'pdf_name': pdf['filename'],
                    'image_paths': image_paths,
                    'status': 'success' if image_paths else 'failed',
                    'pages_converted': len(image_paths)
                })
               
        return results

# Initialize GPU-enabled manager
gpu_conversion_manager = GPUPDFConversionManager()
    

# async def process_image_with_openai(image_path: str) -> Invoice:
#     """Process an image using OpenAI's Vision API asynchronously"""
#     try:
#         # Initialize OpenAI client
#         client = OpenAI(api_key=openai_api_key)
        
#         # Run the synchronous OpenAI call in a thread pool
#         loop = asyncio.get_event_loop()
#         invoice = await loop.run_in_executor(
#             None,
#             partial(process_with_openai_sync, image_path, client)
#         )
        
#         return invoice
        
#     except Exception as e:
#         print(f"Error processing image with OpenAI: {str(e)}")
#         raise

#  revsion 2 ----------------------------------------------------------------------------------------------------------------------------

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

class QueryRequest(BaseModel):
    query: str

class Token(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(..., description="Type of token (e.g., Bearer)")

