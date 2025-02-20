from fastapi import FastAPI,Depends,status,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine, Base,get_db
from response import auth_routes, invoice_routes, processing_routes, query_routes,dev_routes
import os
from database.models import User
from services.auth import get_current_user
from sqlalchemy.orm import Session


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


@app.get("/drop-tables")
async def delete_table(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        return {"status": "success", "message": "Tables dropped successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error dropping tables: {str(e)}"
        )