# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker, declarative_base
# from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession,async_sessionmaker

# SQLALCHEMY_DATABASE_URL = "sqlite:///./database/users.db"
# engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # Async setup
# async_engine = create_async_engine(
#     "sqlite+aiosqlite:///./database/users.db",
#     connect_args={"check_same_thread": False}
# )
# async_session = async_sessionmaker(async_engine, expire_on_commit=False)

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# MySQL connection URL with pymysql
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:Test123!@localhost:3306/invoice"

# Create engine for synchronous connection
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=10, max_overflow=20)

# SessionLocal for synchronous session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base for declarative models
Base = declarative_base()

# Async setup for MySQL with pymysql
async_engine = create_async_engine(
    "mysql+aiomysql://root:Test123!@localhost:3306/invoice",
    connect_args={"check_same_thread": False}
)

# Async session
async_session = async_sessionmaker(async_engine, expire_on_commit=False)

# Function to get synchronous session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()