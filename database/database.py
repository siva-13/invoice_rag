from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker

# Create async engine and session
async_engine = create_async_engine(
    "sqlite+aiosqlite:///./database/users.db",
    connect_args={"check_same_thread": False}
)
async_session = async_sessionmaker(async_engine, expire_on_commit=False)

# Database Configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./database/users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

Base.metadata.create_all(bind=engine)
def get_db():
    db=SessionLocal()
    try:
        yield db
    finally:
        db.close()