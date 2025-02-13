from fastapi import APIRouter
from database.database import Base, engine

router = APIRouter(tags=["Development"])


@router.get("/drop-tables")
async def delete_table():
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, detail=f"Error dropping tables: {str(e)}")
