import uuid
import os
import shutil
from datetime import datetime

def generate_unique_id():
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"USER_{timestamp}_{uuid.uuid4().hex[:8]}"

def delete_all_in_directory(directory):
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)