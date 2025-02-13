# TODO
# Convert into modules
# 
# add validates to avoid same file being processed.
#  
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks,APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.orm import sessionmaker, Session, declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy import text
import asyncio
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy import ForeignKey
from passlib.context import CryptContext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional, List, Any, Dict
from jose import JWTError, jwt
from pdf2image import convert_from_path
from torchvision import transforms
from PIL import Image
from pydantic import BaseModel, Field
# import aiohttp
import json
import base64
from functools import partial
import uuid
import os
import shutil
import time
import numpy as np
import torch
from openai import OpenAI
from response.response import router
from response import response


# Initialize FastAPI app
app = FastAPI(title="Gen AI Invoice API")
app.include_router(response.router)
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

