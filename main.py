# TODO
# Convert into modules
# 
# add validates to avoid same file being processed.
#  
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from responses import auth_routes,delete_routes,invoice_routes,pdf_routes,processing_routes
import database.models
from database.database import engine


# Initialize FastAPI app
app = FastAPI(title="Gen AI Invoice API")
app.include_router(auth_routes.router)
app.include_router(pdf_routes.router)
app.include_router(processing_routes.router)
app.include_router(invoice_routes.router)
app.include_router(delete_routes.router)
database.models.Base.metadata.create_all(bind=engine)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)





















