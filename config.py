from concurrent.futures import ProcessPoolExecutor
import torch
import asyncio
import os
from dotenv import load_dotenv 
from groq import Groq  # ✅ Use OpenAI instead of Groq
from langchain_groq import ChatGroq
from openai import OpenAI
# Load environment variables
load_dotenv()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
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
OPEN_API_KEY= os.getenv("OPEN_API_KEY")
if not OPEN_API_KEY:
    raise ValueError("Missing API Key! Set OPENAI_API_KEY in the .env file.")

# ✅ Initialize OpenAI client
client = OpenAI(api_key=OPEN_API_KEY)
#llm=ChatGroq(model="llama-3.2-90b-vision-preview")
# Initialize Groq LLM with structured output
# llm = ChatGroq(
#     api_key=os.getenv("GROQ_API_KEY"),
#     model="llama3-groq-70b-8192-tool-use-preview",
#     temperature=0
# )