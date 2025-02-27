from database.database import Base,get_db,engine
import os
from fastapi import Depends,HTTPException,status
from database.models import User
from services.auth import get_current_user
from sqlalchemy.orm import Session
import shutil
from config import UPLOAD_DIR,PDF_IMAGE_DIR





# delete all files in directory
def delete_all_in_directory(directory):
    try:
        # Check if the directory exists
        if os.path.exists(directory):
            # Iterate over each item in the directory
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                # If it's a file, delete it
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                # If it's a directory, remove it and its contents
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            print(f"All contents in '{directory}' have been deleted.")
        else:
            print(f"Directory '{directory}' does not exist.")
    except Exception as e:
        print(f"Error deleting contents of '{directory}': {e}")

# Only on development
async def delete_table(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        delete_all_in_directory(UPLOAD_DIR)
        delete_all_in_directory(PDF_IMAGE_DIR)
        return {"status": "success", "message": "Tables dropped successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error dropping tables: {str(e)}"
        )
    