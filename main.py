from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database.database import engine, Base
from response import auth_routes, invoice_routes, processing_routes, query_routes

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
@app.get("/drop-tables")
async def delete_table():
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error dropping tables: {str(e)}")
