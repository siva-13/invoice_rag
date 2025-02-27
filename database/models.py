from pydantic import Field,BaseModel
from sqlalchemy import Column,Integer,String,DateTime,Float,ForeignKey
from sqlalchemy.orm import relationship
from .database import Base
from typing import Optional
from datetime import datetime


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

class QueryRequest(BaseModel):
    query: str

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String(255), unique=True, index=True)
    username = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))
    pdf_files = relationship("PDFFile", back_populates="user")
    invoices = relationship("InvoiceDB", back_populates="user")

class PDFFile(Base):
    __tablename__ = "pdf_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    file_path = Column(String(255))
    upload_time = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String(255), ForeignKey("users.unique_id"))
    user = relationship("User", back_populates="pdf_files")
    invoice = relationship("InvoiceDB", back_populates="pdf_file", uselist=False)

class InvoiceDB(Base):
    __tablename__= "invoices"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.unique_id"), index=True)
    pdf_file_id = Column(Integer, ForeignKey("pdf_files.id"), index=True)
    invoice_number = Column(String(255), index=True)
    seller_name = Column(String(255))
    seller_gstin = Column(String(255), nullable=True)
    date_of_invoice = Column(String(255))
    buyer_order_number = Column(String(255), nullable=True)
    buyer_name = Column(String(255))
    buyer_gstin = Column(String(255), nullable=True)
    number_of_items = Column(Integer, nullable=True)
    total_amount = Column(Float, nullable=True)
    sgst = Column(Float, nullable=True)
    cgst = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="invoices")
    pdf_file = relationship("PDFFile", back_populates="invoice")
    items = relationship("InvoiceItemDB", back_populates="invoice", cascade="all, delete-orphan")

class InvoiceItemDB(Base):
    __tablename__= "invoice_items"
    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), index=True)
    description = Column(String(1000))
    quantity = Column(Integer)
    rate_per_unit = Column(Float)
    amount = Column(Float, nullable=True)
    invoice = relationship("InvoiceDB", back_populates="items")

class ProcessingStatus(Base):
    __tablename__= "processing_status"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), ForeignKey("users.unique_id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_images = Column(Integer)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    status = Column(String(255))
    error_message = Column(String(500), nullable=True)

