from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import User
from services.auth import get_current_user, create_access_token, get_password_hash,get_user,verify_password
from services.utils import generate_unique_id
from config import DEVICE, process_pool, PDF_IMAGE_DIR, UPLOAD_DIR,MAX_WORKERS,BATCH_SIZE,RATE_LIMIT_REQUESTS,RATE_LIMIT_WINDOW,MAX_CONCURRENT_REQUESTS,API_SEMAPHORE,client,api_key


router = APIRouter(tags=["Authentication"])

@router.post("/signup")
async def signup(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    if get_user(db, form_data.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    
    unique_id = generate_unique_id()
    new_user = User(
        username=form_data.username,
        password_hash=get_password_hash(form_data.password),
        unique_id=unique_id
    )
    
    try:
        db.add(new_user)
        db.commit()
        access_token = create_access_token(data={"sub": new_user.username})
        return {
            "message": "User created",
            "username": new_user.username,
            "unique_id": unique_id,
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(500, detail=f"Error creating user: {str(e)}")

@router.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(401, detail="Invalid credentials")
    
    access_token = create_access_token(data={"sub": user.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "unique_id": user.unique_id
    }

@router.get("/users/me")
async def get_user_info(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "unique_id": current_user.unique_id
    }