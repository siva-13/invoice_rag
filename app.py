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

# Update the extract_invoice_data endpoint to include PDF file relationship
@app.post("/extract-invoice-data")
async def extract_invoice_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    openai_api_key: str = Header(..., description="OpenAI API Key")
):
    try:
        # Get user's image directory
        user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
        if not os.path.exists(user_image_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        # Get all image files
        image_files = [
            os.path.join(user_image_dir, f)
            for f in os.listdir(user_image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if not image_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        processed_results = []
        
        # Find related PDF files
        for image_path in image_files:
            # Extract base filename without extension to match with original PDF
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            base_filename = base_filename.split('_page_')[0]  # Remove page number suffix
            
            # Find corresponding PDF file
            pdf_file = db.query(PDFFile).filter(
                PDFFile.user_id == current_user.unique_id,
                PDFFile.filename.like(f"%{base_filename}%")
            ).first()
            
            if not pdf_file:
                continue
            
            # Process image with OpenAI
            result = await process_image_with_openai(image_path, openai_api_key)
            
            if result:
                # Create Invoice instance
                invoice_data = Invoice(**result)
                
                # Save to database with PDF file relationship
                db_invoice = InvoiceDB(
                    user_id=current_user.unique_id,
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
                    raw_response=json.dumps(result)
                )
                
                db.add(db_invoice)
                db.flush()  # Get the ID without committing
                
                # Save invoice items
                for item in invoice_data.item_list:
                    db_item = InvoiceItemDB(
                        invoice_id=db_invoice.id,
                        description=item.description,
                        quantity=item.quantity,
                        rate_per_unit=item.rate_per_unit,
                        amount=item.amount
                    )
                    db.add(db_item)
                
                processed_results.append({
                    "image_path": image_path,
                    "invoice_number": invoice_data.invoice_number,
                    "pdf_filename": pdf_file.filename,
                    "status": "success"
                })
        
        # Commit all changes
        db.commit()
        
        return {
            "status": "success",
            "message": f"Processed {len(processed_results)} images",
            "results": processed_results
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing images: {str(e)}"
        )

# Update the get_invoices endpoint to include relationships
@app.get("/invoices")
async def get_invoices(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        invoices = db.query(InvoiceDB).filter(
            InvoiceDB.user_id == current_user.unique_id
        ).options(
            joinedload(InvoiceDB.items),
            joinedload(InvoiceDB.pdf_file)
        ).all()
        
        results = []
        for invoice in invoices:
            results.append({
                "invoice_number": invoice.invoice_number,
                "seller_name": invoice.seller_name,
                "date_of_invoice": invoice.date_of_invoice,
                "total_amount": invoice.total_amount,
                "pdf_file": {
                    "filename": invoice.pdf_file.filename if invoice.pdf_file else None,
                    "upload_time": invoice.pdf_file.upload_time.isoformat() if invoice.pdf_file else None
                },
                "items": [
                    {
                        "description": item.description,
                        "quantity": item.quantity,
                        "rate_per_unit": item.rate_per_unit,
                        "amount": item.amount
                    }
                    for item in invoice.items
                ]
            })
        
        return {
            "status": "success",
            "invoices": results
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving invoices: {str(e)}"
        )
    


async def process_single_image(image_path: str, session: Session, pdf_file: PDFFile, processing_status_id: int) -> Optional[Invoice]:
    """Process a single image and store the extracted invoice data"""
    try:
        async with API_SEMAPHORE:
            # Encode image
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Call OpenAI API
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "You are an AI assistant specialized in extracting structured data from invoice images. Be very careful with the data extraction process and ensure that the extracted data is accurate. All fields are mandotory",
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
                raw_response=str(response)
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
            status.processed_images += 1
            session.commit()

            return invoice_data

    except Exception as e:
        # Update failed count
        status = session.query(ProcessingStatus).get(processing_status_id)
        status.failed_images += 1
        session.commit()
        
        # Create user directory for failed images if not exists
        user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id, "failed_images")
        os.makedirs(user_image_dir, exist_ok=True)
        
        # Move the failed image to the user's failed images directory
        failed_image_path = os.path.join(user_image_dir, os.path.basename(image_path))
        shutil.move(image_path, failed_image_path)
        
        print(f"Error processing image {image_path}: {str(e)}")
        return None




async def process_invoices_background(
    user_id: str,
    image_paths: list,
    pdf_files: list,
    processing_status_id: int,
    db: Session
):
    """Background task to process invoices"""
    try:
        tasks = []
        for pdf_file in pdf_files:
            base_filename = os.path.splitext(pdf_file.filename)[0]
            pdf_images = [
                path for path in image_paths 
                if path.startswith(base_filename)
            ]
            
            for image_path in pdf_images:
                task = process_single_image(image_path, db, pdf_file, processing_status_id)
                tasks.append(task)

        # Process all images
        await asyncio.gather(*tasks, return_exceptions=True)

        # Update final status
        status = db.query(ProcessingStatus).get(processing_status_id)
        status.status = 'completed'
        status.end_time = datetime.utcnow()
        db.commit()

    except Exception as e:
        # Update status with error
        status = db.query(ProcessingStatus).get(processing_status_id)
        status.status = 'failed'
        status.error_message = str(e)
        status.end_time = datetime.utcnow()
        db.commit()


@app.post("/process-invoices")
async def process_invoices(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Get all image paths for the user
        user_image_dir = os.path.join(PDF_IMAGE_DIR, current_user.unique_id)
        if not os.path.exists(user_image_dir):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        # Get all PDF files for the user
        pdf_files = db.query(PDFFile).filter(
            PDFFile.user_id == current_user.unique_id
        ).all()

        if not pdf_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No PDF files found"
            )

        # Get all relevant image paths
        image_paths = [
            os.path.join(user_image_dir, f) 
            for f in os.listdir(user_image_dir) 
            if f.endswith('.jpg')
        ]

        if not image_paths:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No images found for processing"
            )

        # Create processing status record
        processing_status = ProcessingStatus(
            user_id=current_user.unique_id,
            total_images=len(image_paths),
            status='processing'
        )
        db.add(processing_status)
        db.commit()
        db.refresh(processing_status)

        # Start background processing
        background_tasks.add_task(
            process_invoices_background,
            current_user.unique_id,
            image_paths,
            pdf_files,
            processing_status.id,
            db
        )

        return {
            "status": "processing_started",
            "message": "Invoice processing started in background",
            "total_images": len(image_paths),
            "processing_id": processing_status.id
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting invoice processing: {str(e)}"
        )

@app.get("/processing-status/{processing_id}")
async def get_processing_status(
    processing_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of a processing job"""
    status = db.query(ProcessingStatus).filter(
        ProcessingStatus.id == processing_id,
        ProcessingStatus.user_id == current_user.unique_id
    ).first()

    if not status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Processing job not found"
        )

    return {
        "status": status.status,
        "total_images": status.total_images,
        "processed_images": status.processed_images,
        "failed_images": status.failed_images,
        "start_time": status.start_time,
        "end_time": status.end_time,
        "error_message": status.error_message
    }

#  -------------------------------------------------------------------------------------------------------------------------------------------

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import func
import json

QUERY_PROMPT = """
You are an AI helping to interpret natural language queries about invoices and translate them into structured data.
The database has the following tables and fields:

InvoiceDB:
- invoice_number (str)
- seller_name (str)
- seller_gstin (str)
- date_of_invoice (str)
- buyer_order_number (str)
- buyer_name (str)
- buyer_gstin (str)
- number_of_items (int)
- total_amount (float)
- sgst (float)
- cgst (float)
- created_at (datetime)

InvoiceItemDB:
- description (str)
- quantity (int)
- rate_per_unit (float)
- amount (float)

Based on the user's query, return a JSON object with:
1. fields: List of database fields to query
2. operations: List of operations to perform (sum, average, count, group, find)
3. time_period: Time filter to apply (today, week, month, year, range, or null)
4. filters: Any specific filters to apply
5. group_by: Fields to group by (if any)
6. sort: Sort criteria (if any)
7. natural_response: A template for responding to the user in natural language

Example response:
{
    "fields": ["total_amount"],
    "operations": ["sum"],
    "time_period": "month",
    "filters": {},
    "group_by": null,
    "sort": null,
    "natural_response": "The total amount for all invoices this month is {total_amount}"
}

USER QUERY: {query}
"""

class QueryAnalysis(BaseModel):
    fields: List[str]
    operations: List[str]
    time_period: Optional[str]
    filters: Dict[str, Any]
    group_by: Optional[str]
    sort: Optional[Dict[str, str]]
    natural_response: str

async def analyze_query_with_openai(query: str) -> QueryAnalysis:
    """Use OpenAI to analyze the user's query and determine what information to retrieve"""
    try:
        response = await asyncio.to_thread(
            lambda: client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "user",
                        "content": QUERY_PROMPT.format(query=query)
                    }
                ],
                response_format={"type": "json_object"}
            )
        )
        
        analysis = json.loads(response.choices[0].message.content)
        return QueryAnalysis(**analysis)
        
    except Exception as e:
        print(f"Error in OpenAI query analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error analyzing query"
        )

async def execute_invoice_query(
    analysis: QueryAnalysis,
    user_id: str,
    db: Session
) -> Dict[str, Any]:
    """Execute the analyzed query and return results"""
    try:
        # Start with base query
        query = db.query(InvoiceDB).filter(InvoiceDB.user_id == user_id)
        
        # Apply time period filters
        now = datetime.utcnow()
        if analysis.time_period == "today":
            query = query.filter(InvoiceDB.created_at >= now.replace(hour=0, minute=0, second=0))
        elif analysis.time_period == "week":
            query = query.filter(InvoiceDB.created_at >= now - timedelta(days=7))
        elif analysis.time_period == "month":
            query = query.filter(InvoiceDB.created_at >= now.replace(day=1, hour=0, minute=0, second=0))
        elif analysis.time_period == "year":
            query = query.filter(InvoiceDB.created_at >= now.replace(month=1, day=1, hour=0, minute=0, second=0))
            
        # Apply custom filters
        for field, value in analysis.filters.items():
            if hasattr(InvoiceDB, field):
                query = query.filter(getattr(InvoiceDB, field) == value)
                
        results = {}
        
        # Handle grouping
        if analysis.group_by:
            group_field = getattr(InvoiceDB, analysis.group_by)
            query = query.group_by(group_field)
            
        # Execute operations
        for operation in analysis.operations:
            if operation == "sum":
                for field in analysis.fields:
                    if hasattr(InvoiceDB, field):
                        total = query.with_entities(func.sum(getattr(InvoiceDB, field))).scalar() or 0
                        results[f"total_{field}"] = float(total)
                        
            elif operation == "average":
                for field in analysis.fields:
                    if hasattr(InvoiceDB, field):
                        avg = query.with_entities(func.avg(getattr(InvoiceDB, field))).scalar() or 0
                        results[f"average_{field}"] = float(avg)
                        
            elif operation == "count":
                count = query.count()
                results["count"] = count
                
            elif operation == "group":
                if analysis.group_by:
                    group_results = []
                    grouped_query = (
                        query.with_entities(
                            group_field,
                            func.count(InvoiceDB.id),
                            *[func.sum(getattr(InvoiceDB, field)) for field in analysis.fields if hasattr(InvoiceDB, field)]
                        )
                    )
                    
                    for result in grouped_query.all():
                        group_value = result[0]
                        count = result[1]
                        field_values = result[2:]
                        
                        group_dict = {
                            "group": group_value,
                            "count": count
                        }
                        
                        for field, value in zip(analysis.fields, field_values):
                            group_dict[field] = float(value) if value else 0
                            
                        group_results.append(group_dict)
                        
                    results["grouped_results"] = group_results
        
        # Apply sorting if specified
        if analysis.sort:
            for field, direction in analysis.sort.items():
                if hasattr(InvoiceDB, field):
                    sort_field = getattr(InvoiceDB, field)
                    query = query.order_by(sort_field.desc() if direction == "desc" else sort_field.asc())
        
        return results
        
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing query: {str(e)}"
        )

def format_response(results: Dict[str, Any], template: str) -> str:
    """Format the results using the natural language template"""
    try:
        return template.format(**results)
    except KeyError:
        return str(results)

@app.post("/query-invoices")
async def query_invoices(
    query: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process natural language queries about invoices using OpenAI for interpretation
    
    Example queries:
    - "What is the total amount of all invoices this month?"
    - "Show me the top 5 sellers by total sales"
    - "How many invoices do I have from last week?"
    - "What is the average invoice amount for each buyer?"
    - "List all invoices with GSTIN containing 'XYZ'"
    """
    try:
        # Analyze query using OpenAI
        analysis = await analyze_query_with_openai(query)
        
        # Execute the query
        results = await execute_invoice_query(analysis, current_user.unique_id, db)
        
        # Format natural language response
        natural_response = format_response(results, analysis.natural_response)
        
        return {
            "query": query,
            "natural_response": natural_response,
            "results": results,
            "metadata": {
                "analyzed_fields": analysis.fields,
                "operations_performed": analysis.operations,
                "time_period": analysis.time_period,
                "filters": analysis.filters,
                "group_by": analysis.group_by,
                "sort": analysis.sort
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )
    



from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import and_, or_, func, cast, Float, Integer
from sqlalchemy.orm import Session
import json

class QueryProcessor:
    def __init__(self):
        self.client = OpenAI()

    async def parse_query(self, query: str) -> Dict[str, Any]:
        """Convert natural language query to structured query parameters"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": """Parse the user query into structured parameters for invoice search.
                            Return a JSON object with these possible fields:
                            - invoice_number: exact match
                            - seller_gstin: exact match
                            - buyer_gstin: exact match
                            - date_range: {start_date, end_date}
                            - amount_range: {min_amount, max_amount}
                            - seller_name: partial match
                            - buyer_name: partial match
                            - sort_by: field to sort by
                            - sort_order: asc/desc
                            - aggregation: sum/avg/count
                            - group_by: field to group by"""
                        },
                        {"role": "user", "content": query}
                    ],
                    response_format={ "type": "json_object" }
                )
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error parsing query: {str(e)}"
            )

    def build_query(self, session: Session, params: Dict[str, Any], user_id: str):
        """Build SQLAlchemy query based on parsed parameters"""
        query = session.query(InvoiceDB).filter(InvoiceDB.user_id == user_id)

        # Apply filters
        if params.get('invoice_number'):
            query = query.filter(InvoiceDB.invoice_number == params['invoice_number'])
            
        if params.get('seller_gstin'):
            query = query.filter(InvoiceDB.seller_gstin == params['seller_gstin'])
            
        if params.get('buyer_gstin'):
            query = query.filter(InvoiceDB.buyer_gstin == params['buyer_gstin'])
            
        if params.get('seller_name'):
            query = query.filter(InvoiceDB.seller_name.ilike(f"%{params['seller_name']}%"))
            
        if params.get('buyer_name'):
            query = query.filter(InvoiceDB.buyer_name.ilike(f"%{params['buyer_name']}%"))

        if params.get('date_range'):
            if params['date_range'].get('start_date'):
                query = query.filter(
                    func.date(InvoiceDB.date_of_invoice) >= params['date_range']['start_date']
                )
            if params['date_range'].get('end_date'):
                query = query.filter(
                    func.date(InvoiceDB.date_of_invoice) <= params['date_range']['end_date']
                )

        if params.get('amount_range'):
            if params['amount_range'].get('min_amount'):
                query = query.filter(
                    InvoiceDB.total_amount >= float(params['amount_range']['min_amount'])
                )
            if params['amount_range'].get('max_amount'):
                query = query.filter(
                    InvoiceDB.total_amount <= float(params['amount_range']['max_amount'])
                )

        # Apply sorting
        if params.get('sort_by'):
            sort_field = getattr(InvoiceDB, params['sort_by'], InvoiceDB.created_at)
            if params.get('sort_order') == 'desc':
                query = query.order_by(sort_field.desc())
            else:
                query = query.order_by(sort_field.asc())

        return query

    def process_aggregation(self, session: Session, base_query, params: Dict[str, Any]):
        """Process aggregation requests"""
        if not params.get('aggregation'):
            return base_query.all()

        agg_func = params['aggregation']
        group_by = params.get('group_by')
        
        if group_by:
            group_field = getattr(InvoiceDB, group_by)
            if agg_func == 'count':
                query = session.query(
                    group_field,
                    func.count(InvoiceDB.id).label('count')
                )
            elif agg_func == 'sum':
                query = session.query(
                    group_field,
                    func.sum(InvoiceDB.total_amount).label('sum')
                )
            elif agg_func == 'avg':
                query = session.query(
                    group_field,
                    func.avg(InvoiceDB.total_amount).label('average')
                )
            return query.group_by(group_field).all()
        else:
            if agg_func == 'count':
                return base_query.count()
            elif agg_func == 'sum':
                return base_query.with_entities(
                    func.sum(InvoiceDB.total_amount)
                ).scalar()
            elif agg_func == 'avg':
                return base_query.with_entities(
                    func.avg(InvoiceDB.total_amount)
                ).scalar()

# Initialize query processor
query_processor = QueryProcessor()

@app.post("/query-invoices")
async def query_invoices(
    query: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process natural language queries about invoices and return relevant data
    
    Example queries:
    - "Show me all invoices from last month"
    - "What's the total amount of invoices from seller X?"
    - "Find invoices with amount greater than 1000"
    - "Count invoices by seller"
    - "Find invoice number INV-001"
    - "Show average invoice amount by month"
    """
    try:
        # Parse natural language query
        parsed_params = await query_processor.parse_query(query)
        
        # Build and execute database query
        base_query = query_processor.build_query(db, parsed_params, current_user.unique_id)
        
        # Process any aggregations
        results = query_processor.process_aggregation(db, base_query, parsed_params)
        
        # Format response based on query type
        if isinstance(results, list):
            if hasattr(results[0], '_asdict'):  # For aggregation results
                formatted_results = [dict(row._asdict()) for row in results]
            else:  # For regular invoice results
                formatted_results = []
                for invoice in results:
                    items = db.query(InvoiceItemDB).filter(
                        InvoiceItemDB.invoice_id == invoice.id
                    ).all()
                    
                    formatted_results.append({
                        "id": invoice.id,
                        "invoice_number": invoice.invoice_number,
                        "seller_name": invoice.seller_name,
                        "buyer_name": invoice.buyer_name,
                        "date_of_invoice": invoice.date_of_invoice,
                        "total_amount": invoice.total_amount,
                        "items": [
                            {
                                "description": item.description,
                                "quantity": item.quantity,
                                "rate_per_unit": item.rate_per_unit,
                                "amount": item.amount
                            }
                            for item in items
                        ]
                    })
        else:
            formatted_results = results

        return {
            "query": query,
            "parsed_parameters": parsed_params,
            "results": formatted_results,
            "result_count": len(formatted_results) if isinstance(formatted_results, list) else 1
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )