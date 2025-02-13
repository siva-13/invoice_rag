from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    pdf_files = relationship("PDFFile", back_populates="user")
    invoices = relationship("InvoiceDB", back_populates="user")

class PDFFile(Base):
    __tablename__ = "pdf_files"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    file_path = Column(String)
    upload_time = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, ForeignKey("users.unique_id"))
    user = relationship("User", back_populates="pdf_files")
    invoice = relationship("InvoiceDB", back_populates="pdf_file", uselist=False)

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
    user = relationship("User", back_populates="invoices")
    pdf_file = relationship("PDFFile", back_populates="invoice")
    items = relationship("InvoiceItemDB", back_populates="invoice", cascade="all, delete-orphan")

class InvoiceItemDB(Base):
    __tablename__ = "invoice_items"
    id = Column(Integer, primary_key=True, index=True)
    invoice_id = Column(Integer, ForeignKey("invoices.id"), index=True)
    description = Column(String)
    quantity = Column(Integer)
    rate_per_unit = Column(Float)
    amount = Column(Float, nullable=True)
    invoice = relationship("InvoiceDB", back_populates="items")

class ProcessingStatus(Base):
    __tablename__ = "processing_status"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.unique_id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    total_images = Column(Integer)
    processed_images = Column(Integer, default=0)
    failed_images = Column(Integer, default=0)
    status = Column(String)
    error_message = Column(String, nullable=True)