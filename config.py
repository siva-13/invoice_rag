from concurrent.futures import ProcessPoolExecutor
import torch
import asyncio
import os
from dotenv import load_dotenv
from openai import OpenAI  # ✅ Use OpenAI instead of Groq

# Load environment variables
load_dotenv()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
PDF_IMAGE_DIR = "./pdfs_to_image"
UPLOAD_DIR = "./uploaded_pdfs"
MAX_WORKERS = 10  # Adjust based on your CPU cores
BATCH_SIZE = 10  # Increased batch size for GPU processing
RATE_LIMIT_REQUESTS = 50  # Requests per minute limit for OpenAI API
RATE_LIMIT_WINDOW = 60  # Window in seconds

# Update the semaphore to allow more concurrent operations
MAX_CONCURRENT_REQUESTS = 20  # Increased from 5
API_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Add a process pool for CPU-bound tasks
process_pool = ProcessPoolExecutor(max_workers=4)

# Load OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing API Key! Set OPENAI_API_KEY in the .env file.")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=api_key)
