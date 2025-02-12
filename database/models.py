from sqlalchemy import Column,Integer,String,ForeignKey,DateTime,Float,create_engine
from sqlalchemy.orm import relationship,sessionmaker,declarative_base
from datetime import datetime
from pydantic import BaseModel,Field
from typing import Optional



# Database Configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./database/users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

Base.metadata.create_all(bind=engine)
# Update User model to include relationships
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    
    # Add relationships
    pdf_files = relationship("PDFFile", back_populates="user")
    invoices = relationship("InvoiceDB", back_populates="user")

# Update PDFFile model to include relationships
class PDFFile(Base):
    __tablename__ = "pdf_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, ForeignKey("users.unique_id"))
    
    # Add relationships
    user = relationship("User", back_populates="pdf_files")
    invoice = relationship("InvoiceDB", back_populates="pdf_file", uselist=False)

# Update InvoiceDB model with relationships
class InvoiceDB(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.unique_id"), index=True)
    pdf_file_id = Column(Integer, ForeignKey("pdf_files.id"), index=True)
    invoice_number = Column(String, index=True)
    seller_name = Column(String)
    seller_gstin = Column(String, nullable=True)
    date_of_invoice = Column(String)
    buyer_order_number = Column(String, nullable=True)
    buyer_name = Column(String)
    buyer_gstin = Column(String, nullable=True)
    number_of_items = Column(Integer, nullable=True)
    total_amount = Column(Float, nullable=True)
    sgst = Column(Float, nullable=True)
    cgst = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    raw_response = Column(String)  # Store the complete OpenAI response
    
    # Add relationships
    user = relationship("User", back_populates="invoices")
    pdf_file = relationship("PDFFile", back_populates="invoice")
    items = relationship("InvoiceItemDB", back_populates="invoice", cascade="all, delete-orphan")

# Update InvoiceItemDB model with relationship
class InvoiceItemDB(Base):
    __tablename__ = "invoice_items"

    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), index=True)
    description = Column(String)
    quantity = Column(Integer)
    rate_per_unit = Column(Float)
    amount = Column(Float, nullable=True)
    
    # Add relationship
    invoice = relationship("InvoiceDB", back_populates="items")



# Invoice Model
class Invoice(BaseModel):
    # Invoice Step Model
    class Step(BaseModel):
        description: str = Field(..., description="Description of the item")
        quantity: int = Field(..., description="Quantity of the item")
        rate_per_unit: float = Field(..., description="Rate per unit of the item")
        amount: Optional[float] = Field(None, description="Total amount for the item")
    invoice_number: str = Field(..., description="Unique invoice identifier")
    seller_name: str = Field(..., description="Name of the seller")
    seller_gstin: Optional[str] = Field(None, description="GSTIN of the seller")
    date_of_invoice: str = Field(..., description="Date of the invoice in YYYY-MM-DD format")
    buyer_order_number: Optional[str] = Field(None, description="Order number from the buyer")
    buyer_name: str = Field(..., description="Name of the buyer")
    buyer_gstin: Optional[str] = Field(None, description="GSTIN of the buyer")
    number_of_items: Optional[int] = Field(None, description="Number of items in the order")
    item_list: list[Step]
    total_amount: Optional[float] = Field(None, description="Total amount for the invoice")
    sgst: Optional[float] = Field(None, description="State GST amount")
    cgst: Optional[float] = Field(None, description="Central GST amount")

class Query(BaseModel):
    sqlQuery: str = Field(..., description="SQL query based on user prompt")

class ProcessingStatus(Base):
    __tablename__ = "processing_status"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.unique_id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_images = Column(Integer)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    status = Column(String)  # 'processing', 'completed', 'failed'
    error_message = Column(String, nullable=True)