import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database.database import get_db
from database.models import User
from services.auth import get_current_user, create_access_token, get_password_hash,get_user,verify_password
from datetime import datetime, timedelta
from services.auth import ACCESS_TOKEN_EXPIRE_MINUTES

router=APIRouter()



# API Endpoints
@router.post("/signup")
async def signup(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Check if user already exists
    if get_user(db, form_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Create new user
    unique_id = generate_unique_id()
    new_user = User(
        username=form_data.username,
        password_hash=get_password_hash(form_data.password),
        unique_id=unique_id
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": new_user.username}, 
            expires_delta=access_token_expires
        )
        
        return {
            "message": "User created successfully",
            "username": new_user.username,
            "unique_id": new_user.unique_id,
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating user: {str(e)}"
        )

@router.post("/token")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    # Get user from database
    user = get_user(db, form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=access_token_expires
    )
    
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

# Helper functions
def generate_unique_id():
    """Generate a unique ID based on timestamp and UUID"""
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    random_string = str(uuid.uuid4())[:8]
    return f"USER_{timestamp}_{random_string}"