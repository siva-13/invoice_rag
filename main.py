from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine, Base
from response import auth_routes, invoice_routes, processing_routes, query_routes,dev_routes
import os

app = FastAPI(title="Gen AI Invoice API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Include all routers
app.include_router(auth_routes.router)
app.include_router(invoice_routes.router)
app.include_router(processing_routes.router)
app.include_router(query_routes.router)

# Development-only route
if os.getenv("ENV") == "development":
    app.include_router(dev_routes.router)